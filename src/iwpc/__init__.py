import os

welcome_message = """
Thank you for using
  ___  __        __  ____     ____ 
 |_ _| \ \      / / |  _ \   / ___|
  | |   \ \ /\ / /  | |_) | | |    
  | |    \ V  V /   |  __/  | |___ 
 |___|    \_/\_/    |_|      \____|
                                   
Created by: Jeremy J. H. Wilkinson (jero.wilkinson@gmail.com).
If this tool has been helpful in your research, please consider citing https://arxiv.org/abs/2405.06397.
Disable with the env variable DISABLE_IWPC_WELCOME=1"""

if not os.environ.get('DISABLE_IWPC_WELCOME', default=False):
    print(welcome_message)
