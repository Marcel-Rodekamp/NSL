def main():

    import sys
    import os
    import io
    import yaml

    import argparse
    parser = argparse.ArgumentParser() #help='Process data down to a manageable size by cutting, binning, and bootstrapping.'
    parser.add_argument("yaml", type=str, help="Initialization YAML file")
    parser.add_argument("destination", help="Directory of the ensemble")
    parser.add_argument("--GPU", default=False, action='store_true', help="Using GPU")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Overwrite the directory")
    parser.add_argument("--no-submit", default=False, action='store_true', help="Set up the directory without submitting a job")
    args = parser.parse_args()

    with open(args.yaml) as stream:
        ymlFile = yaml.safe_load(stream)

    dirName = ymlFile["fileIO"]["h5file"].split(".")[0]
    maxCfg = ymlFile["HMC"]["Nconf"]

    print('# Setting up directory')
    if args.overwrite:
        os.system(f'rm -r {args.destination}/{dirName}')

    os.system(f'mkdir -p {args.destination}/{dirName}/output')
    print(f'{args.destination}/{dirName}')

    print('# Generating submission script')
    os.system(f'cp {args.yaml} {args.destination}/{dirName}/.')
    os.system(f'cp submit_chain_batch.sh {args.destination}/{dirName}/.')  # copy environments to local run directory
    os.system(f'cp env.sh {args.destination}/{dirName}/.')  # copy environments to local run directory

    if args.no_submit:
        return

    print('# Submitting job')
    if args.GPU:
        os.system(f"cd {args.destination}/{dirName};./submit_chain_batch.sh {args.yaml} {dirName} {maxCfg} --GPU")
    else:
        os.system(f"cd {args.destination}/{dirName};./submit_chain_batch.sh {args.yaml} {dirName} {maxCfg}")

if __name__ == "__main__":
    main()