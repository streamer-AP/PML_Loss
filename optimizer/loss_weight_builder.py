class Loss_Weight():
    def __init__(self,cfg) -> None:
        if cfg.type=="constant":
            self.weight=1
            self.type="constant"
        elif cfg.type=="from_file":
            with open(cfg.file_path,"r") as f:
                self.weight=[]
                for line in f:
                    self.weight.append(float(line.strip("\n")))
            self.type="from_file"
    
    def __call__(self,epoch):
        if self.type=="constant":
            return self.weight
        elif self.type=="from_file":
            if epoch<len(self.weight):
                return self.weight[epoch]
            else:
                return 1
        else:
            raise NotImplementedError

def loss_weight_builder(cfg):
    return Loss_Weight(cfg)