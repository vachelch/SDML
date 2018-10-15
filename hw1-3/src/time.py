import  re
import numpy as np 
import math, time

dir_name = 't3-doc'
dir = './'

def get_time(dir_name):
	times = []
	for i in range(1, 17500 + 1):
		file = dir_name + '/' + str(i) + '.xml'
		with open(file, 'r') as f:
			html = f.read()
			html = html.replace("\n", "")

		date = re.search(r'<date>(.*)</date>', html).group(1)
		date = date.split()
		for item in date:
			# get year
			if item.isdigit() and int(item) > 31:
				if len(item) == 4:
					year = int(item)
					break
				elif len(item) == 2:
					year = int(item) + 1900
					break

		times.append(year - 1991)

	return times

times = get_time(dir_name)
times = np.array(times)
print(np.max(times), np.min(times))


np.save(dir + 'times.npy', times)


# import  re
# import numpy as np 
# import math, time

# dir_name = 't3-doc'
# dir = '/nfs/Frigga/Vachel/'

# def get_time(dir_name):
# 	times = []
# 	months = {"jan": 1, 'feb' : 2, 'mar': 3, 'apr': 4, 'may' : 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep':9, 'oct':10, "nov": 11 ,'dec': 12}
# 	months_vec = ["jan", 'feb' , 'mar', 'apr', 'may' , 'jun', 'jul', 'aug', 'sep','oct', "nov", 'dec']

# 	for i in range(1, 17500 + 1):
# 		file = dir_name + '/' + str(i) + '.xml'
# 		with open(file, 'r') as f:
# 			html = f.read()
# 			html = html.replace("\n", "")

# 		date = re.search(r'<date>(.*)</date>', html).group(1)
# 		date = date.split()

# 		noyear = 1
# 		nodate = 1
# 		for item in date:
# 			# get year
# 			if noyear and item.isdigit() and int(item) > 31:
# 				if len(item) == 4:
# 					year = int(item)
# 					noyear = 0
# 					break
# 				elif len(item) == 2:
# 					year = int(item) + 1900
# 					noyear = 0

# 			# get month
# 			if item.lower() in months:
# 				month = months[item.lower()]

# 			# get day
# 			if nodate and len(item) <= 2 and int(item) <= 31:
# 				day = int(item)
# 				nodate = 0

# 		t = time.mktime(time.strptime(str(year) + " " + months_vec[month-1] + " " + str(day), "%Y %b %d"))
# 		times.append(t)

# 	return times

# times = get_time(dir_name)
# times = np.array(times)
# print(np.max(times), np.min(times))

# times = times - np.min(times)
# times = np.floor(times * 12 / np.max(times))

# print(np.max(times), np.min(times))
# np.save(dir + 'times.npy', times)








































