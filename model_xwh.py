import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas_bokeh
pandas_bokeh.output_notebook()
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from math import pi
from bokeh.transform import cumsum
from pywaffle import Waffle
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.models import ColumnDataSource, Label, LabelSet
#from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR, ROUND_CEILING



st.set_page_config(
    page_title="Warehouse Kedep",
    layout="wide",)



x_awal=[0.00,0.00,0.00,0.00,0.95,0.95,0.95,0.95]
y_awal=[0.95,0.00,0.95,0.00,0.95,0.00,0.95,0.00]
z_awal=[0.95,0.95,0.00,0.00,0.95,0.95,0.00,0.00]
z_next=[1.35,1.35,0.00,0.00,1.35,1.35,0.00,0.00]



new_raw_data=pd.read_excel('update_SO_final.xlsx')
pilih_blok=list(new_raw_data['zona'].drop_duplicates().sort_index(ascending=True))

colA, colB=st.columns([7,3])
col1, col2 = st.columns([2, 8], gap="small")
col3, col4, col5 = st.columns([3,6,6], gap="small")
col6, col7, col8 = st.columns([3,6,6], gap="small")


new_raw_data['col_pal'] = new_raw_data['item'].apply(lambda x: '#07c6a6' if x == '1500ML' else '#d73f4f' if x == '600ML' else '#175fbc')
new_raw_data['std'] = new_raw_data['item'].apply(lambda x: 70 if x == '1500ML' else 40 if x == '600ML' else 65).astype(int)
#new_raw_data['Marks'] = df['Marks'].round(decimals=1)
#new_raw_data['pallet']= new_raw_data['cart']/new_raw_data['std'].round(decimals=0)
#new_raw_data['pallet']= new_raw_data['pallet'].map(lambda x: (Decimal('0.1'), ROUND_CEILING))

new_raw_data['pallet']= np.ceil(new_raw_data['cart']/new_raw_data['std'])
new_raw_data['batch_new']=new_raw_data['batch'].astype(str)
new_raw_data['batch_new']=new_raw_data['batch'].astype(str)

all_stock=new_raw_data.groupby(["item"])["cart"].sum().reset_index(name='jumlah').sort_values(['item'], ascending=True) 
all_stock_1=new_raw_data.groupby(["item"])["cart"].sum().reset_index(name='jumlah').sort_values(['item'], ascending=True) 


all_stock_1.loc[:, "new_sum"] = all_stock_1["jumlah"].map('{:,d}'.format)
all_stock_1['new_sum']=all_stock_1['new_sum'].str.replace(",",".")
all_stock_1.drop(['jumlah'], axis='columns', inplace=True)

all_stock_2=all_stock_1[['item','new_sum']]

all_stock_2['code'] = all_stock_2['new_sum'].str.rjust(20)
all_stock_2['len'] = all_stock_2['new_sum'].str.len()




length_item = max(len(x) for x in all_stock_1['item'])
length_newsum= max(len(x) for x in all_stock_1['new_sum'])


waffle_wrn = list(all_stock['item'].apply(lambda x: '#07c6a6' if x == '1500ML' else '#d73f4f' if x == '600ML' else '#175fbc'))
waffle_4=dict(all_stock.values)
waffle_5=dict(all_stock_1.values)


#botol='<a href="https://www.flaticon.com/free-icons/water-bottle" title="water bottle icons">Water bottle icons created by andinur - Flaticon</a>'
#<a href="https://www.flaticon.com/free-icons/plastic" title="plastic icons">Plastic icons created by monkik - Flaticon</a>

#st.table(all_stock_1)
#st.text(length_item)
#st.text(length_newsum)


#link=  <1 class="fi fi-rs-water-bottle"></i>


with colA:

    plt.figure(
  FigureClass=Waffle,
  rows=10,
  columns=25,
  values=waffle_4,
  colors= waffle_wrn,
  labels=["{0} : {1} Karton".format(k, v) for k, v in waffle_5.items()],

  legend={'loc': 'upper left', 'bbox_to_anchor': (1.05, 1),'fontsize':10, },
  figsize=(9, 6),
  icons= 'fi fi-ss-water-bottle' , icon_size=14, icon_legend=True,
  title= {'label': 'Komposisi dan Jumlah Produk di Gudang Kedep', 'loc': 'left', 'fontsize':12, 'weight':'bold', })



    st.pyplot(plt)

    st.divider()



with col1:
    pilihan=st.selectbox(label="**Detail Produk di Blok:**",options= pilih_blok)




filter_baru=new_raw_data[new_raw_data['zona'] == pilihan]


groupby_filter= filter_baru.groupby(["item"])["cart"].sum().reset_index(name='sum').sort_values(['item'], ascending=True) 
groupby_filter['angle'] = groupby_filter['sum']/groupby_filter['sum'].sum() * 2*pi
groupby_filter['color'] = groupby_filter['item'].apply(lambda x: '#07c6a6' if x == '1500ML' else '#d73f4f' if x == '600ML' else '#175fbc')
groupby_filter['prosentase']=100*(groupby_filter['sum']/groupby_filter['sum'].sum())
groupby_filter.loc[:, "new_sum"] = groupby_filter["sum"].map('{:,d}'.format)
groupby_filter['new_sum']=groupby_filter['new_sum'].astype(str)
groupby_filter['new_sum']=groupby_filter['new_sum'].str.replace(",",".")


sep = []
for i in range(len(groupby_filter.index)):
            sep.append(':  ')

#st.text(sep)

groupby_filter['legend'] = groupby_filter['item'] + " : " + groupby_filter['new_sum'].astype(str) +" Karton"      
       

pv = figure(plot_height=300, plot_width=250, frame_width=200,title= f"Jumlah Produk di Blok {pilihan}", toolbar_location="above",
           tools="hover",tooltips="@item: @prosentase{0.2f} %", x_range=(-.6, .6))

pv.annular_wedge(x=0, y=100,  inner_radius=0.17, outer_radius=0.46, direction="anticlock", 
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend="legend", source=groupby_filter)

#st.table(groupby_filter)   
#st.text(groupby_filter.dtypes)


pv.axis.axis_label=None
pv.axis.visible=False
pv.grid.grid_line_color = None
pv.legend.location = "center"
pv.add_layout(pv.legend[0], 'below')
pv.legend.label_text_font_size = '10pt'



with col3:
    st.bokeh_chart(pv, use_container_width=True)

#repeat_row sejumlah pallet
df_new=pd.DataFrame(filter_baru.values.repeat(filter_baru.pallet, axis=0), columns=filter_baru.columns)
df_new['sequence']=df_new.groupby('loc').cumcount()+1
df_new['cart']=df_new['cart'].astype(int)
df_new['std']=df_new['std'].astype(int)
df_new['cumm']=df_new.groupby(['row', 'item', 'batch'])['std'].cumsum()
df_new['sisa']=df_new['cart']-df_new['cumm']
df_new['qty'] = np.where(df_new['sisa']>=0, df_new['std'] , df_new['std']+df_new['sisa'])
df_new['level'] = np.where(df_new['item']== '1500ML',2,3)
df_new['one']=df_new.apply(lambda x:'%s / %s' % (x['item'],x['batch']),axis=1)

df_new=df_new.sort_values(by=['row','qty'], ascending=[True, False])
df_new['urut']=df_new.groupby(['zona', 'row']).cumcount()+1
df_new['class'] = np.where(df_new['level']==3, np.ceil(df_new['urut']/3)-1 , np.ceil(df_new['urut']/2)-1)
df_new['id_sort']=df_new['loc'].astype(str)+"-"+df_new['class'].astype(str)


stock_blok=df_new.groupby(['item', 'batch'])['qty'].sum().reset_index(name='Jumlah')
stock_blok['batch']=stock_blok['batch'].astype(str)
stock_prod=stock_blok['item'].drop_duplicates().tolist()


for i, row in df_new.iterrows():
    hasil1 = ''
    if (row['row']==1 and row['level']==3):
        hasil1 = x_awal
    elif (row['row']>1 and row['level']==3):
        hasil1 = [x + row['row'] - 1 for x in x_awal] 
    elif (row['row']==1 and row['level']==2):
        hasil1 = x_awal   
    else:  
        hasil1 = [x + row['row'] - 1 for x in x_awal] 
    df_new.loc[i,['x1','x2','x3','x4','x5','x6','x7','x8'] ] = hasil1


for n, row in df_new.iterrows():
    hasil2 = ''
    if (row['level']==3 and row['urut']<=3):
        hasil2 = y_awal
    elif (row['level']==3 and row['urut']>3):
        hasil2 = [x + row['class'] for x in y_awal]
    elif (row['level']==2 and row['urut']<=2):
        hasil2 = y_awal 
    else:    
        hasil2 = [x + row['class'] for x in y_awal]
    df_new.loc[n,['y1','y2','y3','y4','y5','y6','y7','y8'] ] = hasil2


for m, row in df_new.iterrows():
    hasil2 = ''
    if (row ['level']==3 and row['urut'] in (1,4,7,10,13,16,19,22,25,28,31,34)):
        hasil2 = z_awal
    elif (row['level']==3 and row['urut'] in (2,5,8,11,14,17,20,23,26,29,32,35)):
        hasil2 = [x + 1 for x in z_awal] 
    elif (row['level']==3 and row['urut'] in (3,6,9,12,15,18,21,24,27,30,33,36)):
        hasil2 = [x + 2 for x in z_awal]
    elif (row['level']==2 and row['urut'] in (1,3,5,7,9,11,13,15,17,19)):
        hasil2 = z_next
    else:
        hasil2 = [x + 1.4 for x in z_next]   
    df_new.loc[m,['z1','z2','z3','z4','z5','z6','z7','z8'] ] = hasil2


for i, row in df_new.iterrows():
    hasil1 = ''
    if (row['item']=='600ML' and row ['qty']==40):
        hasil1 = 'sunsetdark'
    elif (row['item']=='1500ML' and row ['qty']==70):    
        hasil1 = 'tealgrn'
    elif (row['item']=='330ML' and row ['qty']==65):    
        hasil1 = 'ice'    
    else:
        hasil1= 'oxy'
    df_new.at[i,'color']  = hasil1

x_gb=df_new[['x1','x2','x3','x4','x5','x6','x7','x8']].values.tolist()
y_gb=df_new[['y1','y2','y3','y4','y5','y6','y7','y8']].values.tolist()
z_gb=df_new[['z1','z2','z3','z4','z5','z6','z7','z8']].values.tolist()
v_gb=df_new[['qty','qty','qty','qty','qty','qty','qty','qty']].values.tolist()
hover_text=df_new[['one','one','one','one','one','one','one' ]].values.tolist()
pick_warna = df_new['color'].tolist()
lokasi = df_new[['loc','loc','loc','loc','loc','loc','loc','loc']].values.tolist()



model_3d = go.Figure()
i = 0
while i < len(x_gb):
    model_3d.add_trace(go.Isosurface(
    x=x_gb[i],
    y=y_gb[i],
    z=z_gb[i],
    value=v_gb[i],
    showscale=False,
    opacity=1,
    colorscale=pick_warna[i],
    text=lokasi[i],
    hovertext=hover_text[i],
    hovertemplate=
            "<b>Lokasi:</b> %{text}" +
            "<br><b>Item / Batch:</b> %{hovertext}" +
            "<br><b>Qty:</b> %{value:d3-format} <extra></extra>" 

       ))
    i += 1  # Update kondisi iterasi

 
model_3d.update_layout(
    hoverlabel=dict(
    bgcolor="white",
    font_size=12,
    font_family="Helvetica"),
                )

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=3, z=3))

model_3d.update_layout(scene_camera=camera)

loc_textA=list(df_new['loc'].drop_duplicates())
loc_valA=list(df_new['row'].drop_duplicates())


df_anyar=df_new.drop_duplicates(subset='id_sort', keep="first")
row_zona=df_anyar['row'].max()-df_anyar['row'].min()+1
row_zona_min=df_anyar['row'].min()
plt_on_fl=len(df_anyar)
depth_zona=df_anyar['class'].max()+1
cap=int(row_zona*depth_zona)
sst=round((len(df_anyar)/cap*100),1)


model_3d.update_layout(title=f'Visualisasi Penyimpanan Produk di Blok {pilihan}',title_font_family='Helvetica',
                       title_xanchor='left',title_yanchor='top', title_y=1, title_x=0.08, title_font_size=13)
model_3d.update_layout(autosize=False,width=500,height=450, margin=dict(t=40, l=3, b=0.1, r=0.5),)
model_3d.update_layout(scene = dict(xaxis = dict(title= f"Blok {pilihan}" ,
                                                ticktext= loc_textA, tickvals= loc_valA,
                                                     showticklabels=True),
                    yaxis = dict(title='Depth', showticklabels=True),
                    zaxis = dict(title='Level', showticklabels=False), ))

config = {'scrollZoom': False}
model_3d.update_layout(scene_aspectmode='data')


citation = Label(x=70, y=70, x_units='screen', y_units='screen',
                 text='Collected by Luke C. 2016-04-01',
                 border_line_color='black', background_fill_color='white')

model_3d.add_annotation(x=0.8, y=0.8,
            text=f"Okupansi Blok {pilihan} : {plt_on_fl} pallet dari kapasitas {cap} pallet posisi",
            showarrow=False, font=dict(
                color="black",
                size=12),
            yshift=0)


with col5:
    st.plotly_chart(model_3d, config=config)



#st.table(df_new)
#st.text(cap)
#st.text(sst)



hapus=filter_baru.drop_duplicates(subset=['col_pal'],keep='first').sort_values('item', ascending=True)
warna=hapus['col_pal'].tolist()


group_X = filter_baru.groupby(['item','batch_new'])['cart'].sum().reset_index(name='cart')
group_X['item_batch_new']=group_X['item']+group_X['batch_new']

group_A = group_X.groupby(['item','batch_new'])


index_cmap = factor_cmap('item_batch_new', palette=warna, factors=sorted(filter_baru.item.unique()), end=1)

#st.table(group_A)

pm = figure(width=325, height=300, title=f"Jumlah Karton Produk Per Batch di Blok {pilihan}",
           x_range=group_A, toolbar_location=None, tooltips=[("item/batch", "@item_batch_new"), ("Jumlah", "@cart_mean")])


pm.vbar(x='item_batch_new', top='cart_mean', width=1, source=group_A,
       line_color="white", fill_color=index_cmap)


pm.y_range.start = 0
pm.x_range.range_padding = 0.05
pm.xgrid.grid_line_color = None
pm.xaxis.major_label_orientation = 1.2
pm.outline_line_color = None
pm.frame_width=450
pm.plot_height=350
pm.plot_width=450

with col4:
    st.bokeh_chart(pm)



#i = 0

#while (i < len(stock_prod)):
  
#    item1=stock_prod[i]
#    batch_pr=stock_blok[stock_blok['item']== stock_prod[i]]['batch'].tolist()
#    counts=stock_blok[stock_blok['item']== stock_prod[i]]['Jumlah'].tolist()


#    pA = figure(x_range=batch_pr, width=500,height=300, title=f"Jumlah Karton {item1} di Blok {pilihan}",
#           toolbar_location=None, tools="", tooltips=[("Batch", "@x"), ("Jumlah", "@top")])

#    pA.vbar(x=batch_pr, top=counts, width=0.7, fill_color='#3288bd')

#    pA.xgrid.grid_line_color = None
#    pA.y_range.start = 0


    #st.bokeh_chart(pA)
#    i += 1  # Update kondisi iterasi


