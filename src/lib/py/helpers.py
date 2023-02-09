import sys

def commandline_args(defaults):
    keys = list(defaults.keys())

    helpstring = f"Syntax: {sys.argv[0]} "
    for k in keys:
        if(defaults[k] == "<required>"): helpstring += f" <{k}>"
        else:                            helpstring += f" [{k}:{defaults[k]}]"

    # Do we just want to know how to call this script?
    if(len(sys.argv)==2):
        if(sys.argv[1] == "--help" or sys.argv[1] == "-h"):
            print(helpstring, file=sys.stderr)
            sys.exit(1)

    # Fill in parameters from commandline and defaults, converting to appropriate types
    args = []
    for i in range(len(keys)):
        default = defaults[keys[i]]
        if(len(sys.argv)<=i+1):
            if(default == "<required>"):
                print(helpstring, file=sys.stderr)
                sys.exit(1)
            else:
                args.append(default)
        else:                
            args.append(type(default)(sys.argv[i+1]))

    return args
