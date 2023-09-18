import torch

def optimizer_builder(args,model_without_ddp):
    params = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if p.requires_grad],
            "lr": args.lr,
        },
    ]
    if args.type.lower()=="sgd":
        return torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.type.lower()=="adam":
        return torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    elif args.type.lower()=="adamw":
        return torch.optim.AdamW(params, args.lr, weight_decay=args.weight_decay)