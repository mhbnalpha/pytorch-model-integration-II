"""All Configureables should go here

Returns:
    None
"""

from utils import get_full_path

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(override=True)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


class Printable:
    def _get_attributes(self):
        attributes = [
            attr
            for attr in dir(self)
            if not attr.startswith("__") and not callable(getattr(self, attr))
        ]

        return attributes

    def _get_dict(self):
        attributes = self._get_attributes()
        d = {i: getattr(self, i) for i in attributes}
        return d

    def __repr__(self):
        d = self._get_dict()
        s = str(d)
        return s

    def __iter__(self):
        for k, v in self._get_dict().items():
            yield k, v


class Config(Printable):
    DB = "centrox"
    COLLECTION = "cloths"
    AWS_ACCESS_KEY_ID = AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY = AWS_SECRET_ACCESS_KEY
    INDEX_BUCKET = "imagesearch-indexes"
    MONGO_HOST = (
        "a8fe1206c092511eab2b60a1835f0bf3-2138289406.ap-south-1.elb.amazonaws.com"
    )
    MONGO_USERNAME = "admin"
    MONGO_PASSWORD = "Qwerty12"
    MONGO_AUTHSOURCE = "centrox"
    MONGO_DATABASE = "centrox"
    REDIS_HOST = "174.138.121.187"
    REDIS_PASSWORD = "Qwerty12#$"
    REDIS_PORT = "6379"
    AI_URL = "https://ai.centrox.xyz/api/predict/imagevec"
    CLOTH_COUNT_URI = "https://www.centrox.xyz/api/cloth/getcount"
    GENDER_HOST = "139.59.54.148"
    IMAGE_RETRIEVAL_HOST = "139.59.49.109"
    IMAGE_RETRIEVAL_MODELNAME = "resnet_encoder_inter"
    IMAGE_RETRIEVAL_VERSION = 1
    IMAGE_SIZE = 128
    EMBEDDING_SIZE = 128
    CLOSEST_TOP_K = 10
    BATCH_SIZE = 50
    INDEX_DIR = get_full_path("../data")
