import argparse,json,sys

def main(argv=None):
    parser=argparse.ArgumentParser(prog="wheatpheno",description="Wheat trait modelling, DSM/DTM height estimation and spatial maps")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("verify",help="Check fixed model hashes and software compatibility")
    for name in ["demo","review"]:
        p=sub.add_parser(name);p.add_argument("--output",required=True)
    p=sub.add_parser("predict");p.add_argument("--input",required=True);p.add_argument("--output",required=True);p.add_argument("--target",choices=["PH","AGB","Yield"],required=True)
    p.add_argument("--model", help="Local refit joblib with matching JSON metadata")
    p.add_argument("--trust-model", action="store_true")
    p=sub.add_parser("pack");p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--model-metadata", help="Exported refit JSON defining required rasters")
    p=sub.add_parser("height");p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p=sub.add_parser("evaluate");p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p=sub.add_parser("extract");p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p.add_argument("--profile",choices=["standard","historical"],default="standard")
    p.add_argument("--block-size",type=int,default=10)
    p=sub.add_parser("map");p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--targets",nargs="+",choices=["PH","AGB","Yield"],default=None)
    p.add_argument("--fill-distance",type=float,default=0.,help="Display-only nearest fill radius in metres (default zero)")
    p.add_argument("--model", help="Local refit joblib with matching JSON metadata")
    p.add_argument("--trust-model", action="store_true")
    args=parser.parse_args(argv)
    try:
        from .pipeline import predict_csv,map_scene,run_demo
        from .audit import verify,review
        if args.command=="verify":result=verify()
        elif args.command=="review":result=review(args.output)
        elif args.command=="demo":result=run_demo(args.output)
        elif args.command=="map":
            targets = args.targets
            if targets is None:
                if args.model:
                    from .deployment import read_metadata
                    targets = [read_metadata(args.model)["target"]]
                else:
                    targets = ["PH", "AGB", "Yield"]
            result=map_scene(args.config,args.output,targets,args.fill_distance,args.model,args.trust_model)
        elif args.command=="height":
            from .height import estimate_height
            result=estimate_height(args.config,args.output)
        elif args.command=="pack":
            from .prepare import pack_rasters
            result=pack_rasters(args.config,args.output,args.model_metadata)
        elif args.command=="extract":
            from .pipeline import extract_archive
            result=extract_archive(args.input,args.output,args.profile,args.block_size)
        elif args.command=="evaluate":
            from .training import evaluate_table
            result=evaluate_table(args.config,args.output)
        else:result=predict_csv(args.input,args.output,args.target,args.model,args.trust_model)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 1 if result.get("status")=="failed" else 0
    except (ValueError,TypeError,FileNotFoundError,FileExistsError,RuntimeError) as error:
        print(f"WheatPheno: {error}",file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
