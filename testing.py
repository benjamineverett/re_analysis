house_nums = (1200,1500)
blocks = {}

if house_nums[0] > 99:
    remainder = house_nums[0] % 100
    block = house_nums[0] - remainder

elif house_nums[0] < 100:
    block = 0

while block < house_nums[-1]:
    # blocks[block] = list(range(block, block + 100))
    blocks[block] = {'even': [x for x in list(range(block, block + 100)) if x % 2 == 0]}
    blocks[block].update({'odd': [x for x in list(range(block, block + 100)) if x % 2 != 0]})
    block += 100

print(blocks)
