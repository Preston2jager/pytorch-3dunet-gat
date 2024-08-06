import json

class Data_meta:
    _instance = None

    def __init__(self):
        self.hashcode = None
        self.Points_Dir = None

    @staticmethod
    def get_instance():
        if Data_meta._instance is None:
            Data_meta._instance = Data_meta()
        return Data_meta._instance

    def set_hashcode(self, hashcode):
        self.hashcode = hashcode

    def set_Points_Dir(self, points_dir):
        self.Points_Dir = points_dir

    def to_dict(self):
        return {
            'hashcode': self.hashcode,
            'Points_Dir': self.Points_Dir
        }

    @staticmethod
    def from_dict(data):
        instance = Data_meta.get_instance()
        instance.hashcode = data.get('hashcode')
        instance.Points_Dir = data.get('Points_Dir')
        return instance

    def save_to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def load_from_json(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return Data_meta.from_dict(data)
