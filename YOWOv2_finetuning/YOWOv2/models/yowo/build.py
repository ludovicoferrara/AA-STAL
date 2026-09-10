import torch
from .yowo import YOWO
from .loss import build_criterion


# build YOWO detector
def build_yowo(args,
                d_cfg,
                m_cfg, 
                device, 
                num_classes=80, 
                trainable=False,
                resume=None):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))

    # build YOWO
    model = YOWO(
        cfg = m_cfg,
        device = device,
        num_classes = num_classes,
        conf_thresh = args.conf_thresh,
        nms_thresh = args.nms_thresh,
        topk = args.topk,
        trainable = trainable,
        multi_hot = d_cfg['multi_hot'],
        )

    if trainable:
        # Freeze backbone
        if args.freeze_backbone_2d:
            print('Freeze 2D Backbone ...')
            for m in model.backbone_2d.parameters():
                m.requires_grad = False
        if args.freeze_backbone_3d:
            print('Freeze 3D Backbone ...')
            for m in model.backbone_3d.parameters():
                m.requires_grad = False
            
        # keep training       
        # keep training       
        if resume is not None:
            print('Caricamento pesi da: ', resume)
            checkpoint = torch.load(resume, map_location='cpu', weights_only=False)            
            # Se il checkpoint contiene la chiave 'model', estraila. 
            # Altrimenti assumi che il file sia direttamente il state_dict.
            if "model" in checkpoint:
                checkpoint_state_dict = checkpoint.pop("model")
            else:
                checkpoint_state_dict = checkpoint
            
            # MODIFICA: Filtro per il mismatch delle classi
            model_state_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in checkpoint_state_dict.items() 
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            
            if len(pretrained_dict) != len(checkpoint_state_dict):
                print("ATTENZIONE: Mismatch delle dimensioni rilevato. Scarto i tensori incompatibili (es. class head).")
            
            model_state_dict.update(pretrained_dict)
            model.load_state_dict(model_state_dict, strict=False)
            print(f"Caricati {len(pretrained_dict)} su {len(model_state_dict)} tensori.")
            
        # build criterion
        criterion = build_criterion(
            args, d_cfg['train_size'], num_classes, d_cfg['multi_hot'])
    
    else:
        criterion = None
                        
    return model, criterion
