import os


for app in ['coffea', 'rdf']:
    if app == 'rdf':
        options = ['O1', 'O2', 'O3']
    else: options = [app]
    for option in options:
        #for nfiles in range(50, 1000, 200):
        for nfiles in [3750]:
            folder = f'benchmarks/{nfiles}/{option}'
            os.makedirs(folder, exist_ok=True)  
            for ncores in list(range(32,129,32)):

                file = f'{ncores}'
                file = f'{folder}/{file}'
                com = f'touch {file}'
                os.system(com)
                print(com)
                com = f'/usr/bin/time python {app}_ttbar.py --ncores {ncores} --nfiles {nfiles} > {file} 2>&1'
                if app == 'rdf': com = f'env EXTRA_CLING_ARGS=-{option} ' + com
                print(com)
                os.system(com)
