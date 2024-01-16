from functools import partial

def get_coords(x, y, z = None):
    if z == None:
        return x, y, 0
    return x, y, z

def main():
    x = 1
    y = 2
    z = 3

    func = partial(get_coords, z = z)
    result = func(x, y)
    print(result)

main()