from typing import Generic
import json


class ConfigDict(dict):

    def __getattr__(self, key: Generic) -> Generic:
        return self[key]

    def __setattr__(self, key: Generic, value: Generic) -> None:
        self[key] = value

    def copy(self):
        new = ConfigDict()

        for key, value in self.items():
            new[key] = value.copy() if isinstance(value, ConfigDict) else value

        return new

    def update(self, other) -> None:
        for key, value in other.items():
            if isinstance(value, ConfigDict) and isinstance(self[key], ConfigDict):
                self[key].update(value)
            else:
                self[key] = value

    def save_json(self, file: str) -> None:
        with open(file, 'w') as f:
            json.dump(self, f)

    @staticmethod
    def load_json(file: str):
        with open(file, 'r') as f:
            config = json.load(f, object_hook=lambda d: ConfigDict(**d))  # object hook maps dict to ConfigDict

        return ConfigDict(config)