from src.modules import describe_dataframe, generate_code


data_path = "data/sample.csv"
data_info = describe_dataframe(data_path)

user_request = "データの概要について教えて"
response = generate_code(
    data_info=data_info,
    user_request=user_request,
)
program = response.content
print(program.code)
