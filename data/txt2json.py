import json

if __name__ == '__main__':
	txt_name = raw_input('txt file name: ')
	json_name = raw_input('json file name: ')
	
	txt_file = open(txt_name+'.txt', 'r')
	json_file = open(json_name+'.json', 'a')
	
	label = []
	data = []
	while True:
		txt = txt_file.readline().split()
		if not txt:
			break
		label.append(int(txt[0]))
		data.append(txt[1:])
	txt_file.close()

	for i in range(len(label)):
		json_file.write(json.dumps(
							{'label': label[i], 'text': data[i]},
							encoding='utf-8',
							ensure_ascii=False, 
							sort_keys=True)
		)
		json_file.write('\n')
	json_file.close()
	