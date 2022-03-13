from .BaseModel import TwoTowerBaseModel



class TwoTowerModel(TwoTowerBaseModel):
    def __init__(self, manager, newsEncoder, userEncoder):
        super().__init__(manager, name="-".join(["TwoTower", newsEncoder.name, userEncoder.name]))
        self.newsEncoder = newsEncoder
        self.userEncoder = userEncoder

