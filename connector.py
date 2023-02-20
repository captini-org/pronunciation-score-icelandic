__doc__ = """connector.py

Subscribe to relevant topics from RMQ channel and publish results

"""
import argparse
from datetime import date, datetime
import os
from pathlib import Path
import wave
import json
import sys
import typing
from typing import Literal, Iterable

import pika

from captiniscore import PronunciationScorer
from captinialign import AlignOneFunction

InputTopic = Literal["SYNC_SPEECH_INPUT"]
OutputTopic = Literal["PRONUNCIATION_SCORE", "PRONUNCIATION_ALIGNMENT"]


def json_serialize(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


class Consumer:
    def __init__(self, args, callback):
        self._connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=args.rabbitmq_host)
        )
        self._channel = self._connection.channel()
        self._channel.exchange_declare(
            exchange=args.rabbitmq_exchange, exchange_type="direct"
        )

        self._result = self._channel.queue_declare("", exclusive=True)
        self._queue_name = self._result.method.queue

        topics: Iterable[InputTopic] = typing.get_args(InputTopic)
        for topic in topics:
            self._channel.queue_bind(
                exchange=args.rabbitmq_exchange,
                queue=self._queue_name,
                routing_key=topic,
            )

        self._channel.basic_consume(
            queue=self._queue_name,
            on_message_callback=callback if callback else self._default_callback,
            auto_ack=True,
        )

    def _default_callback(self, ch, method, properties, body) -> None:
        print(" [x] %r:%r" % (method.routing_key, body))

    def start_consuming(self) -> None:
        self._channel.start_consuming()


class Producer:
    def __init__(self, args):
        self._connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=args.rabbitmq_host)
        )
        self._channel = self._connection.channel()
        self._exchange = args.rabbitmq_exchange
        self._channel.exchange_declare(exchange=self._exchange, exchange_type="direct")

    def publish(self, message: dict, topic: OutputTopic) -> None:
        self._channel.basic_publish(
            exchange=self._exchange,
            routing_key=topic,
            body=json.dumps(message, default=json_serialize),
        )

    def close(self) -> None:
        self._connection.close()


def main():
    # files provided as examples to use for score demo
    demo_info_file = "./demo_recording_data.tsv"

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wav-dir", type=str, default="./demo-wav/")
    parser.add_argument(
        "--reference-feat-dir",
        type=str,
        default="./reference-feats_w2v2-base_layer-6/",
    )
    parser.add_argument(
        "--speech-featurizer-path",
        type=str,
        default="facebook/wav2vec2-base",
        help="""\
        Speech embedding model and layer must match the pre-computed reference sets.
        This featurizer path loads the model from huggingface, which occasionally has
        connection problems.  For more stable use, download the models from
        https://huggingface.co/facebook/wav2vec2-base and change the path to local
        directory such as './models/facebook/wav2vec2-base'
        """,
    )
    parser.add_argument("--speech-featurizer-layer", type=int, default=6)
    parser.add_argument("--rabbitmq-exchange", type=str, default="captini")
    parser.add_argument("--rabbitmq-host", type=str, default="rabbitmq")
    args = parser.parse_args()

    scorer = PronunciationScorer(
        wav_dir=args.wav_dir,
        reference_feat_dir=args.reference_feat_dir,
        model_path=args.speech_featurizer_path,
        model_layer=args.speech_featurizer_layer,
    )

    def score_it(args, exercise_text, exercise_id, speaker_id, recording_id) -> dict:
        """Score a single utterance.

        Audio file expected to exists at Path(wav_dir, speaker_id, f"{speaker_id}-{rec_id}.wav")

        Args:
          exercise_text

          exercise_id

          speaker_id

          recording_id

        """
        # TODO(rkjaran): The recording ID should probably include both the exercise_id
        #   and an attempt ID. We ignore that for now.
        rec_id = recording_id

        wav_path = Path(args.wav_dir, speaker_id, f"{speaker_id}-{rec_id}.wav")
        with wave.open(str(wav_path), "r") as wav_f:
            rec_duration = wav_f.getnframes() / wav_f.getframerate()

        aligner = AlignOneFunction(
            exercise_text,
            speaker_id,
            rec_id,
            str(rec_duration),
            wav_dir=args.wav_dir,
        )
        word_aligns, phone_aligns = aligner.align()

        word_scores, phone_scores = scorer.score_one(
            exercise_id, speaker_id, rec_id, word_aligns, phone_aligns
        )

        return {
            "word_scores": word_scores,
            "phone_scores": phone_scores,
            "word_aligns": word_aligns,
            "phone_aligns": phone_aligns,
        }

    producer = Producer(args)

    def callback(ch, method, properties, body) -> None:
        if method.routing_key == "SYNC_SPEECH_INPUT":
            # TODO(rkjaran): Properly deserialize timestamp and deadline
            # TODO(rkjaran): Parse and honor deadline and timestamp
            msg = json.loads(body)
            session_id = msg.get("session_id", "UNKNOWN_SESSION")
            text_id = msg["text_id"]
            exercise_text = msg["text"]
            speaker_id = msg["speaker_id"]

            # TODO(rkjaran): Audio file expected to exists at Path(wav_dir, speaker_id, f"{speaker_id}-{rec_id}.wav").
            #   So either we have the IO manager/backend enforce that or we move things around here.
            recording_id = msg["recording_id"]

            scores = score_it(
                args,
                exercise_text=exercise_text,
                exercise_id=text_id,
                speaker_id=speaker_id,
                recording_id=recording_id,
            )

            ret_pronunciation_score = {
                "timestamp": datetime.now(),
                "speaker_id": speaker_id,
                "session_id": session_id,
                "text_id": text_id,
                "score": -1.0,
                "word_scores": [
                    {
                        "word": word,
                        "score": score,
                    }
                    for (word, score) in scores["word_scores"]
                ],
            }

            producer.publish(ret_pronunciation_score, "PRONUNCIATION_SCORE")

            ret_pronunciation_alignment = {
                "timestamp": datetime.now(),
                "speaker_id": speaker_id,
                "session_id": session_id,
                "text_id": text_id,
                "alignment": [
                    {
                        "word": score_tup[0],
                        "phones": [phone for (phone, score) in score_tup[1]],
                        "phone_scores": [score for (phone, score) in score_tup[1]],
                    }
                    for score_tup in scores["phone_scores"]
                ],
            }

            producer.publish(ret_pronunciation_alignment, "PRONUNCIATION_ALIGNMENT")

        else:
            print(" [!] unknown message type: %r:%r" % (method.routing_key, body))

    consumer = Consumer(args, callback=callback)
    consumer.start_consuming()


if __name__ == "__main__":
    main()
