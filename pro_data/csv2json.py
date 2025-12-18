import csv
import json
from collections import defaultdict

# 读取CSV文件并按表名分组
def read_csv_columns(csv_file):
    """读取CSV文件，将字段信息按表名分组"""
    table_columns = defaultdict(list)
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            table_name = row['表名']
            field = row['字段']
            meaning = row['含义']
            remark = row.get('备注', '')  # 备注可能为空
            
            # 组合字段信息：字段+含义+备注
            if remark:
                column_info = f"{field}: {meaning} ({remark})"
            else:
                column_info = f"{field}: {meaning}"
            
            table_columns[table_name].append(column_info)
    
    return table_columns

# 更新JSON字典
def update_json_dictionary(json_file, table_columns, output_file):
    """更新JSON字典的column_description字段"""
    
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    
    # 统计更新情况
    updated_count = 0
    not_found_count = 0
    
    # 遍历JSON中的每个表
    for table_entry in data_dict:
        table_name = table_entry['table_name']
        
        # 如果该表在CSV中有字段信息
        if table_name in table_columns:
            # 将所有字段信息用分号连接
            column_description = '; '.join(table_columns[table_name])
            table_entry['column_description'] = column_description
            updated_count += 1
        else:
            not_found_count += 1
    
    # 保存更新后的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 处理完成！")
    print(f"📊 更新了 {updated_count} 个表的字段描述")
    print(f"⚠️  {not_found_count} 个表在CSV中未找到字段信息")
    print(f"💾 结果已保存到: {output_file}")
    
    return data_dict

# 主函数
def main():
    # 文件路径
    csv_file = 'data/singabi_meta_columns.csv'
    json_file = 'data/singabi_data_dictionary.json'
    output_file = 'data/singabi_data_dictionary_updated.json'
    
    print("🚀 开始处理...")
    
    # 步骤1: 读取CSV文件
    print("📖 读取CSV文件...")
    table_columns = read_csv_columns(csv_file)
    print(f"   找到 {len(table_columns)} 个表的字段信息")
    
    # 步骤2: 更新JSON字典
    print("📝 更新JSON字典...")
    update_json_dictionary(json_file, table_columns, output_file)
    
    # 显示示例
    print("\n📋 更新示例:")
    for table_name in list(table_columns.keys())[:2]:  # 显示前2个表
        print(f"\n表名: {table_name}")
        print(f"字段描述: {'; '.join(table_columns[table_name][:3])}...")

if __name__ == "__main__":
    main()