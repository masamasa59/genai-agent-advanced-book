from src.modules.describe_dataframe import describe_dataframe
from src.modules.generate_code import generate_code


data_path = "data/sample.csv"
data_info = describe_dataframe(data_path)

user_request = "データの概要について教えて"
response = generate_code(
    data_info=data_info,
    user_request=user_request,
)
program = response.content
print(program.code)
