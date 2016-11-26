from sklearn.cross_validation import train_test_split

filename = "letter-recognition.data"
datafile = open(filename, 'r')
x = []
y = []
for line in datafile.readlines():
    if line[-1] == '\n':
        line = line[:-1]
    splited_line = line.split(',')
    for i in range(len(splited_line)):
        if splited_line == 'None':
            splited_line[i] = -1
    x.append(splited_line[1:])
    y.append(str(ord(splited_line[0]) - ord('A')))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

def extenddata(datax, datay):
    newx = []
    newy = []
    for i in range(len(datax)):
        for j in range(26):
            tmp = list(datax[i])
            tmp.append(str(j))
            newx.append(tmp)
            if str(j) == datay[i]:
                newy.append('1')
            else:
                newy.append('0')
    return [newx, newy]

def savefile(path, datax, datay):
    newfile = open(path, 'w')
    for i in range(len(datax)):
        tmp = list(datax[i])
        tmp.append(datay[i])
        tmp = ','.join(tmp)
        newfile.write(tmp + '\n')

x_train_ext, y_train_ext = extenddata(x_train, y_train)
x_test_ext, y_test_ext = extenddata(x_test, y_test)

savefile("train.data", x_train, y_train)
savefile("test.data", x_test, y_test)
savefile("train_ext.data", x_train_ext, y_train_ext)
savefile("test_ext.data", x_test_ext, y_test_ext)
