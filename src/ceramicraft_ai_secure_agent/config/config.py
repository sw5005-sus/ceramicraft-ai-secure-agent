import yaml
from pydantic import BaseModel
from typing import Dict


class RedisConfig(BaseModel):
    host: str
    port: int


class KafkaConfig(BaseModel):
    bootstrap_servers: str
    group_id: str


class Config(BaseModel):
    redis: RedisConfig
    kafka: KafkaConfig


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
