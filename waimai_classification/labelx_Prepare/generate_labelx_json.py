#coding=utf-8
import json
import sys
config = '{"url":"","type":"image","multiple":0,"label":[]}'
def gene_labelx_result_json_file(image_list_file,url_prefix=None):
    save_labelx_result_json_file = image_list_file+'_labelx.json'
    with open(image_list_file,'r') as f,open(save_labelx_result_json_file,'w') as w_f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            if url_prefix :
                url = url_prefix + line
            else:
                url = line
            line_dict = eval(config)
            line_dict['url'] = url
            w_f.write(json.dumps(line_dict)+'\n')

def main():
    url_file = sys.argv[1]
    gene_labelx_result_json_file(url_file)
    pass
if __name__ == '__main__':
    main()