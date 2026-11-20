"""CLI entrypoint: python main.py --mode tiger|pipe|landmark|model"""
import argparse
import sys


def main():
    parser=argparse.ArgumentParser(description='MediaPipe gesture-driven AR effect demos')
    parser.add_argument('--mode',choices=('tiger','pipe','landmark','model'),default='landmark',help='Select a demo')
    parser.add_argument('--camera',type=int,default=0,help='Webcam device number, default 0')
    parser.add_argument('--model-path',default=None,help='Keras .h5 path, model mode only')
    parser.add_argument('--confidence',type=float,default=0.8,help='Model mode confidence threshold')
    args=parser.parse_args()
    try:
        if args.mode=='tiger':
            from gesture_ar.tiger import run
            run(args.camera)
        elif args.mode=='pipe':
            from gesture_ar.pipe import run
            run(args.camera)
        elif args.mode=='landmark':
            from gesture_ar.landmark import run
            run(args.camera)
        else:
            from gesture_ar.classifier import run
            run(args.camera,args.model_path,args.confidence)
    except (RuntimeError,FileNotFoundError,ValueError) as exc:
        print(f'[ERROR] {exc}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
