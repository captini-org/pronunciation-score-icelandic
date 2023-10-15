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
from captinialign import AlignOneFunction, makeAlign
from captinifeedback import FeedbackConverter
import librosa
import logging

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
            #pika.ConnectionParameters(host=args.rabbitmq_host)
            pika.ConnectionParameters(host=args.rabbitmq_host, heartbeat=1000)
        )
        self._channel = self._connection.channel()
        self._channel.exchange_declare(
            #exchange=args.rabbitmq_exchange, exchange_type="direct"
            exchange='captini', exchange_type="direct"
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
            pika.ConnectionParameters(host=args.rabbitmq_host, heartbeat=1000)
        )
        self._channel = self._connection.channel()
        # self._exchange = args.rabbitmq_exchange
        self._exchange = 'captini'
        self._channel.exchange_declare(exchange=self._exchange, exchange_type="direct")

    def publish(self, message: dict, topic: OutputTopic) -> None:
        self._channel.basic_publish(
            exchange='captini',
            routing_key='PRONUNCIATION_ALIGNMENT',
            body=json.dumps(message),
        )
        body=json.dumps(message, default=json_serialize)
    def close(self) -> None:
        self._connection.close()
def display_as_json(score_output):
    task_feedback = score_output['task_feedback']
    word_feedback = score_output['word_feedback']
    phone_feedback = score_output['phone_feedback']
    
    feedback_json = {
        'task_feedback': task_feedback,
        'word_feedback': [],
    }
    if(task_feedback == 0):
        word_data = {
            'word': 'Please record the audio again and ensure it matches the provided text.',
            'word_score': 0,
            'phone_feedback': [],
        }
        feedback_json['word_feedback'].append(word_data)
    else:
        for w_s, p_s in zip(word_feedback, phone_feedback):
            assert w_s[0] == p_s[0]
            word_data = {
                'word': w_s[0],
                'word_score': w_s[1],
                'phone_feedback': [],
            }
            
            for i in range(len(p_s[1])):
                phone_data = {
                    'phone': p_s[1][i][0],
                    'phone_score': p_s[1][i][1],
                }
                word_data['phone_feedback'].append(phone_data)
            
            feedback_json['word_feedback'].append(word_data)
        
    # Convert the JSON structure to a string
    feedback_json_str = json.dumps(feedback_json, indent=4)
    # Return the JSON structure as a string
    return feedback_json


def main():
    # Define constants for converting pronunciation scores
    # to user feedback
    binary_threshold = -0.005
    lower_bound_100 = -0.1
    upper_bound_100 = 0.0

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
        default="./task_models_w2v2-IS-1000h/",
    )
    parser.add_argument(
        "--speech-featurizer-path",
        type=str,
        default="carlosdanielhernandezmena/wav2vec2-large-xlsr-53-icelandic-ep10-1000h",
        help="""\
        Speech embedding model and layer must match the pre-computed reference sets.
        This featurizer path loads the model from huggingface, which occasionally has
        connection problems.  For more stable use, download the models from
        https://huggingface.co/facebook/wav2vec2-base and change the path to local
        directory such as './models/facebook/wav2vec2-base'
        """,
    )
    parser.add_argument("--speech-featurizer-layer", type=int, default=8)
    parser.add_argument("--rabbitmq-exchange", type=str, default="captini")
    parser.add_argument("--rabbitmq-host", type=str, default="rabbitmq")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="DEBUG")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)


    scorer = PronunciationScorer(
        reference_feat_dir=args.reference_feat_dir,
        model_path=args.speech_featurizer_path,
        model_layer=args.speech_featurizer_layer,
    )
    # FeedbackConverter new module to process scores into user feedback
    fb = FeedbackConverter(binary_threshold, lower_bound_100, upper_bound_100)
    def score_it(args, exercise_text, exercise_id, speaker_id, recording_id, path) -> dict:

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

        wav_path = Path("/wavs", path)
        with wave.open(str(wav_path), "r") as wav_f:
            rec_duration = wav_f.getnframes() / wav_f.getframerate()


        word_aligns, phone_aligns = makeAlign(
        exercise_text,
        wav_path,
        rec_duration,
        speaker_id
        )
        logging.info("word_aligns: %s, phone_aligns: %s", word_aligns, phone_aligns)

        if(len(word_aligns)>0):
            task_text, task_model = scorer.task_scorer(exercise_id)
            if(task_text!= ''):        
                word_scores, phone_scores = scorer.score_one(
                task_model,str(wav_path), word_aligns, phone_aligns
                )
                task_feedback, word_feedback, phone_feedback = fb.convert(
                word_scores,
                phone_scores)

                score_output = {'task_feedback': task_feedback,
                            'word_feedback': word_feedback, 'phone_feedback': phone_feedback,
                            'word_scores': word_scores, 'phone_scores': phone_scores,
                            'word_aligns': word_aligns, 'phone_aligns': phone_aligns}
            
            else:
                score_output = {'task_feedback': 100,
                            'word_feedback': ['Please record the audio again and ensure it matches the provided text.'], 'phone_feedback':[],
                            'word_scores': [], 'phone_scores': [],
                            'word_aligns': [], 'phone_aligns': []}
        else:
            score_output = {'task_feedback': 0,
            'word_feedback': [], 'phone_feedback':[],
            'word_scores': [], 'phone_scores': [],
            'word_aligns': [], 'phone_aligns': []}
        #display(score_output)
        score_feeback = display_as_json(score_output)
        return score_feeback
    publisher = Producer(args)
    def callback(ch, method, properties, body) -> None:
        if method.routing_key == "SYNC_SPEECH_INPUT":
            # TODO(rkjaran): Properly deserialize timestamp and deadline
            # TODO(rkjaran): Parse and honor deadline and timestamp
            msg = json.loads(body)
            logging.info("Handling message: %s", msg)
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
                path=msg["audio_path"],
            )
            ret_pronunciation_alignment = {
                "speaker_id": speaker_id,
                "session_id": session_id,
                "text_id": text_id,
                "recording_id": recording_id,

            }
            ret_pronunciation_alignment["score"]= scores

            publisher.publish(ret_pronunciation_alignment,'PRONUNCIATION_ALIGNMENT')
        else:
            print(" [!] unknown message type: %r:%r" % (method.routing_key, body))
    consumer = Consumer(args, callback=callback)
    consumer.start_consuming()

if __name__ == "__main__":
    main()
