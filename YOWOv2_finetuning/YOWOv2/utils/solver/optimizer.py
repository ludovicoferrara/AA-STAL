import torch
from torch import optim


def build_optimizer(cfg, model, base_lr=0.0, resume=None):
    print('==============================')
    print('Optimizer: {}'.format(cfg['optimizer']))
    print('--momentum: {}'.format(cfg['momentum']))
    print('--weight_decay: {}'.format(cfg['weight_decay']))

    if cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=base_lr,
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])

    elif cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=base_lr,
            eight_decay=cfg['weight_decay'])
                                
    elif cfg['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=base_lr,
            weight_decay=cfg['weight_decay'])
          
    start_epoch = 0
    if resume is not None:
        print('Verifica stato ottimizzatore in: ', resume)
        # 1. Risolve l'errore di PyTorch 2.6
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        
        # 2. Evita il crash se si caricano pesi pre-addestrati o architetture modificate
        if "optimizer" in checkpoint and "epoch" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                start_epoch = checkpoint["epoch"]
                print(f"Stato ottimizzatore ripristinato. Ripartenza dall'epoca {start_epoch}.")
            except Exception as e:
                print("Mismatch nell'ottimizzatore (cambio numero classi). Inizializzazione pulita.")
                start_epoch = 0
        else:
            print("Nessuno stato precedente trovato. Inizializzazione ottimizzatore pulita.")
                                
    return optimizer, start_epoch                    
                                
    return optimizer, start_epoch
