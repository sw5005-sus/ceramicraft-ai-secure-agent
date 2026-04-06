import os

import yaml
from pydantic import BaseModel


class RedisConfig(BaseModel):
    host: str
    port: int


class KafkaConfig(BaseModel):
    bootstrap_servers: str
    group_id: str


class MysqlConfig(BaseModel):
    host: str
    port: int
    username: str
    password: str
    database: str
    max_connect_size: int


class Config(BaseModel):
    redis: RedisConfig
    kafka: KafkaConfig
    mysql: MysqlConfig


def load_config() -> Config:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_file_dir, "config.yaml")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


system_config = load_config()
