def print_dividing_line(title="", l=100, c='='):
    if title == "":
        print(c*l)
    else:
        print('{s:{c}^{l}}'.format(s=" {} ".format(title), l=l, c=c))
        
def log_dividing_line(logger, title="", l=100, c='='):
    if title == "":
        logger.info(c*l)
    else:
        logger.info('{s:{c}^{l}}'.format(s=" {} ".format(title), l=l, c=c))