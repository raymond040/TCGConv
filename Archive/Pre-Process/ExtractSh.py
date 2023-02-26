import scipy.io
import pandas as pd
mat = scipy.io.loadmat('/home/svu/e0407728/My_FYP/TCGConv/Dataset/YelpCHI/YelpChi.mat') 

net_rur = pd.DataFrame(mat['net_rur']).to_csv('/home/svu/e0407728/My_FYP/TCGConv/Dataset/YelpCHI/net_rur.csv')
net_rtr = pd.DataFrame(mat['net_rtr']).to_csv('/home/svu/e0407728/My_FYP/TCGConv/Dataset/YelpCHI/net_rtr.csv')
net_rsr = pd.DataFrame(mat['net_rsr']).to_csv('/home/svu/e0407728/My_FYP/TCGConv/Dataset/YelpCHI/net_rsr.csv')
features = pd.DataFrame(mat['features']).to_csv('/home/svu/e0407728/My_FYP/TCGConv/Dataset/YelpCHI/features.csv')
homo = pd.DataFrame(mat['homo']).to_csv('/home/svu/e0407728/My_FYP/TCGConv/Dataset/YelpCHI/homo.csv')
label = pd.DataFrame(mat['label']).to_csv('/home/svu/e0407728/My_FYP/TCGConv/Dataset/YelpCHI/label.csv')

label
net_rur
net_rtr
net_rsr
features
homo
