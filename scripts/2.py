from safe_rl.pg.EnvCD2 import ValueController

if __name__ == '__main__':

    cuss = ['3476', '2259', '2821', '3358', '3386', '3427', '1458', '2997']
    addresss = ['0606', '1019', '1019', '0606', '0606', '0606', '0606', '1021']
    bids = [246, 287, 291, 226, 300, 234, 300, 277]

    for i in range(len(cuss)):
        cus, address, bid = cuss[i], addresss[i], bids[i]
        w = False
        if i >= 4:
            w = True
        env = ValueController(basedir='/home/leping_zhang/rl/torch1/data/', ad=cus, bid=bid, dat=address,
                              w=False, otherwarm=w)
        total = env.show_bound()
        with open("/home/leping_zhang/saferl/result.log", encoding="utf-8",mode="a") as file:
            file.write(cus+"------------\n")
            file.write(str(round(total, 2))+'\n')