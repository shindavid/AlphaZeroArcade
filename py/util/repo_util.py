import os


class Repo:
    _instance = None

    @staticmethod
    def instance():
        if Repo._instance is None:
            Repo._instance = Repo()
        return Repo._instance

    def __init__(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        marker = os.path.join(root, 'REPO_ROOT_MARKER')
        assert os.path.isfile(marker), f'Missing repo root marker file: {marker}'
        self._root = root

    @staticmethod
    def root():
        return cls.instance()._root
    
    @staticmethod
    def c4_games():
        return os.path.join(Repo.root(), 'c4_games')
    
    @staticmethod
    def c4_model():
        return os.path.join(Repo.root(), 'c4_model.pt')
