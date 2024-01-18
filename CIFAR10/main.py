import arg_parse
import train
import test

def main(args):
    if args.choice == "train":
        train.train(args)
    elif args.choice == "test":
        test.test(args)
    else:
        print("Must give value into choice")
        return

if __name__ == "__main__":
    parser = arg_parse.get_parse()
    args = parser.parse_args()
    main(args)