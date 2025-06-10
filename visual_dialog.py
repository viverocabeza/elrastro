# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:07:22 2025

@author: dvive
"""

import os
import streamlit as st
import pandas as pd
from seaborn import diverging_palette
import numpy as np
from itertools import repeat
import zipfile
# pd.set_option("styler.render.max_elements", 302324)

st.set_page_config(page_icon = '🤑', page_title = 'El Rastro',
                   layout = "wide", initial_sidebar_state = "collapsed")
@st.cache_data
def load_data():
    folder = st.session_state['folder']
    with zipfile.zipfile(folder) as z:
        file = pd.read_excel(z.open('file.xlsx'), index_col = 0)
        competitors = pd.read_excel(z.open('competitors.xlsx'), index_col = 0)
        competitors = competitors[competitors.index.isin(file.index)]
        file.loc[competitors.index, 'Competitors'] = competitors['Competitors']
        file = file[file.last_update > '2024-12-31']
        #file_old = pd.read_excel(folder + 'file_20250405.xlsx', index_col = 0)
        IS = pd.read_csv(z.open('IS.csv'), index_col = 0).dropna()
        IS.Value = IS.Value.astype(float)
        BS = pd.read_csv(z.open('BS.csv'), index_col = 0).dropna()
        BS.Value = BS.Value.astype(float)
    return file, IS, BS #, file_old

@st.cache_resource
def load_favs():
    folder = st.session_state['folder']
    return list(pd.read_excel(folder + 'favs.xlsx', index_col = 0).index)
    


@st.dialog("Data folder")
def load_folder():
    folder = st.file_uploader('Upload data').read()
    if folder is not None:
        st.write(folder)
    if st.button("Submit"):
        with zipfile.zipfile(folder) as z:
            file = pd.read_excel(z.open('file.xlsx'), index_col = 0)
            competitors = pd.read_excel(z.open('competitors.xlsx'), index_col = 0)
            competitors = competitors[competitors.index.isin(file.index)]
            file.loc[competitors.index, 'Competitors'] = competitors['Competitors']
            file = file[file.last_update > '2024-12-31']
            #file_old = pd.read_excel(folder + 'file_20250405.xlsx', index_col = 0)
            IS = pd.read_csv(z.open('IS.csv'), index_col = 0).dropna()
            IS.Value = IS.Value.astype(float)
            BS = pd.read_csv(z.open('BS.csv'), index_col = 0).dropna()
            BS.Value = BS.Value.astype(float)
        #st.session_state['folder'] = folder
        st.rerun()
        return file, IS, BS
    return

@st.fragment
def filtering():    
    st.session_state['DB'] = st.multiselect("DB", file.DB.unique()) #, default = 'US')
    st.session_state['F'] = st.checkbox("Only Favorites?")
    exclude = st.checkbox("Exclude sector?")
    st.session_state['sector'] = st.multiselect("Sector", file.Sector.unique())
    if st.session_state['sector']:
        st.session_state['ind'] = st.multiselect("Sub-industry", file[file.Sector.isin(st.session_state['sector'])]['Sub-industry'].unique())
    else:
        st.session_state['ind'] = st.multiselect("Sub-industry", file['Sub-industry'].unique())

    if exclude and st.session_state['sector'] and not st.session_state['ind']:
        st.session_state['sector'] = list(filter(lambda x: x not in st.session_state['sector'], 
                                                 file.Sector.unique()))

    elif exclude and st.session_state['ind']:
        st.session_state['sector'] = file.Sector.unique()
        st.session_state['ind'] = list(filter(lambda x: x not in st.session_state['ind'], 
                                                 file['Sub-industry'].unique()))

        
    st.session_state['var'] = st.multiselect("FILTER var: ", 
                                             ['Cap', 'ROA', 'margin', 'Op_margin', 
                                              'cash', 'rev_g', 'EV/Rev', 'PS', 'PB'])

    st.session_state['limits'] = dict()
    for var in st.session_state['var']:
        st.session_state['limits'][var] = st.text_input(var, placeholder = '(** **) Filter ' + var)

    # st.session_state['add_cols'] = st.multiselect("ADD cols: ", 
    #                                          ['PEn', 'Asset_g', 'PTB'])
    st.session_state['compare'] = st.checkbox("Compare?")
    if st.button('Update'):
        filter_file.clear()
        format_df.clear()
        st.rerun()
        

def filter_by_row():  
    with col2:      
        if len(st.session_state.file['selection']['rows']) > 0:
            ticker = file_.iloc[st.session_state.file['selection']['rows'][0]].name
            cols_is = ['Total Revenue', 'Normalized Income', 'Net Income Common Stockholders', 
                       'Total Premiums Earned', 'EBIT', 'Research & Development']
            mapping = {'Total Revenue' : 'Revenue', 
                       'Normalized Income' : 'Norm.Inc', 
                       'Net Income Common Stockholders' : 'NI', 
                       'Total Premiums Earned' : 'Premiums'}
            cols_bs = ['Invested Capital', 'Total Assets', 'Common Stock Equity']
            # filt = file_.loc[ticker]
            st.checkbox('Favorite', value = file.loc[ticker].name in st.session_state['favs'], 
                                  on_change = save_favs, key = 'favs_update')
            with st.expander(ticker + ':  ' + str(file.loc[ticker, 'descr'])[:280] + '...'):
                try:
                    st.write('...' + str(file.loc[ticker, 'descr'])[280:])
                except:
                    pass
            IS_filt = IS_[IS_.Security == ticker].reset_index()
            IS_filt = IS_filt.loc[IS_filt.Breakdown.isin(cols_is)]
            BS_filt = BS_[BS_.Security == ticker].reset_index()
            BS_filt = BS_filt.loc[BS_filt.Breakdown.isin(cols_bs)]
            # IS_filt = pd.pivot_table(IS_filt, values='Value', index = 'Date',
            #               columns=['Breakdown'], aggfunc="mean").reset_index()
            # # st.write(filt['descr'][:180] + '(...)')ç
            IS_chart = IS_filt.loc[IS_filt.Breakdown.isin(cols_is[:-2])]
            IS_chart.Breakdown = IS_chart.Breakdown.map(lambda x: mapping[x])
            st.bar_chart(IS_chart, x="Date", y='Value', color = 'Breakdown', 
                         stack=False, height = 350)

            try:
                IS_filt = pd.pivot_table(IS_filt, values='Value', index = 'Date',
                              columns=['Breakdown'], aggfunc="mean")
                BS_filt = pd.pivot_table(BS_filt, values='Value', index = 'Date',
                              columns=['Breakdown'], aggfunc="mean")
                
                df = pd.DataFrame(IS_filt.get('EBIT', np.nan) /  
                                  BS_filt.get('Invested Capital', np.nan), 
                                  columns = ['ROIC']) * 100
                df['ROA'] =  (IS_filt.get('Normalized Income', np.nan) /  
                                  BS_filt.get('Total Assets', np.nan)) * 100

                df['PE'] =  (file.loc[ticker, 'Cap'] / IS_filt.get('Normalized Income', np.nan) * 1e3)
                try:
                    df['Asset_g'] =  (BS_filt.get('Total Assets', np.nan) /  
                                      BS_filt.get('Total Assets', np.nan).shift(1) - 1) * 100
                except:
                    pass
                df['ROE'] =  (IS_filt.get('Normalized Income', np.nan) /  
                                  BS_filt.get('Common Stock Equity', np.nan)) * 100
            
                # st.write(df.dropna(how = 'all').style.format('{:.1f}').to_html(), 
                #          unsafe_allow_html=True)
                st.dataframe(df.dropna(how = 'all').iloc[::-1].style.format('{:.1f}'))
                # s_df = 
            except:
                st.write('No accounting data')
    # with col1:
    #     if len(st.session_state.file['selection']['rows']) > 0:
    #         st.dataframe(filter_by_name(file, ticker))
        


    #st.write(DB)


@st.cache_data
def filter_file(file):
    cols = ['Name', 'Sector', 'Sub-industry', 'Cap', 'EV', 'cash', 'margin', 
            'Op_margin', 'ROA', 'rev_g', 'NI_g', 'div', 'PEt', 'PEf', 'PS', 
            'EV/rev','EV/EBITDA', 'PB', 'ROE', 'CF', '5D', '1M', 'YTD', 'Competitors']

    filt = file.DB.isin(st.session_state['DB'])
    if st.session_state['ind']:   
        filt &= file['Sub-industry'].isin(st.session_state['ind'])
        # if len(st.session_state['ind']) == 1:
        #     cols.remove('Sub-industry')
    elif st.session_state['sector']:
        filt &= file['Sector'].isin(st.session_state['sector'])
    # if len(st.session_state['sector']) == 1:
    #     cols.remove('Sector')
    if st.session_state['F']:
        filt &= file.index.isin(st.session_state['favs'])
    for var in st.session_state['var']:
        limits = st.session_state['limits'][var].split(' ')
        if len(limits) == 1:    
            filt &= file[var] > float(limits[0])
        elif limits[0]:
            filt &= (file[var] > float(limits[0])) & (file[var] < float(limits[1]))
        else:
            filt &= file[var] < float(limits[1])

    file_ = file[filt][cols]
    IS_ = IS[IS.Security.isin(file_.index)]
    BS_ = BS[BS.Security.isin(file_.index)]
    file_ = add_columns(file_, IS_, BS_)
    return file_, IS_, BS_ 

def add_columns(file, IS, BS):
    file['cash'] = (file['Cap'] - file['EV']) / file['Cap']
    file['PCF'] = file['Cap'] / file['CF']
    
    if st.session_state['compare']:
        file_old_ = file_old.loc[list(set(file_old.index).intersection(file.index))]
        index = file_old_.index
        file['ΔPE'] = file.loc[index].PEt / file_old_.loc[index].PEt - 1
        file['ΔCap'] = file.loc[index].Cap / file_old_.loc[index].Cap - 1

    try: ##ROIC
        # IS_filt = IS[IS.Security.isin(file.index) & 
        #               IS.Date.str.contains('TTM')].loc['EBIT'].set_index('Security').Value
        # RD = IS[IS.Security.isin(file.index) & 
        #               IS.Date.str.contains('TTM')].loc['Research & Development'].set_index('Security').Value
        # BS_filt = BS[BS.Security.isin(file.index) & 
        #               BS.Date.str.contains('2023')].loc['Invested Capital'].set_index('Security').Value
        
        IS_filt = IS[IS.Security.isin(file.index)].loc['EBIT'].set_index('Security').Value
        RD = IS[IS.Security.isin(file.index)].loc['Research & Development'].set_index('Security')
        BS_filt = BS[BS.Security.isin(file.index)].loc['Invested Capital'].set_index('Security').Value
        
        def ROIC(ticker):
            try:
                return float((np.median(IS_filt[ticker]) + np.median(RD.get(ticker, 0)))/ np.median(BS_filt[ticker]))
            except:
                return np.nan
        file['ROIC'] = file.index.map(lambda x: ROIC(x))
        
    except:
        file['ROIC'] = np.nan
        
    try:
        filt = BS.Security.isin(file.index) 
        EQ = BS[filt].loc['Common Stock Equity'].set_index('Security')
        assets = BS[filt].loc['Total Assets'].set_index('Security')
        def EQ_to_assets(ticker):
            try:
                return EQ.loc[ticker].iloc[-1].Value / assets.loc[ticker].iloc[-1].Value
            except:
                return np.nan
        file['E/A'] = file.index.map(lambda x: EQ_to_assets(x))
    except:
        file['E/A'] = np.nan


    return file

    
def filter_by_name(file, tickers):
    if len(tickers) == 0:
        return file
    tickers = [ticker.split(' - ')[0] for ticker in tickers]
    if len(tickers) == 1:
        # st.write([tickers[0]] + file.loc[tickers[0], 'Competitors'].split(','))
        tickers = [tickers[0]] + str(file.loc[tickers[0], 'Competitors']).split(',')
        # return pd.DataFrame(file.loc[tickers[0]]).T
    return file.loc[file.index.isin(tickers)]

def color_margin(x):
    c1 = 'background-color: #da776d' 
    c2 = ''
    c3 = 'background-color: #6da38b'
    mask = x.margin >= 0.9 * x.Op_margin
    mask2 = (x.margin < 0.4 * x.Op_margin) & (x.margin > 0)
    df = pd.DataFrame(c2, index = x.index, columns = x.columns)
    df.loc[mask, 'margin'] = c1
    df.loc[mask2, 'margin'] = c3
    return df

# @st.fragment
def save_favs():
    
    global favs
    ticker = file_.iloc[st.session_state.file['selection']['rows'][0]].name
    if st.session_state.favs_update:
        st.session_state['favs'] += [ticker]
    else:
        st.session_state['favs'].remove(ticker)
    folder = st.session_state['folder']
    pd.Series(index = st.session_state['favs']).to_excel(folder + 'favs.xlsx')
    # load_favs.clear()

@st.cache_resource
def format_df(file_):
    cols = ['Name', 'Sector', 'Sub-industry', 'Cap', 'cash', 'margin', 'Op_margin', 
            'ROA', 'ROIC', 'rev_g', 'div', 'PEt', 'PEf', 'PCF', 'PS', 'EV/rev', 
            'EV/EBITDA', 'PB', 'ROE', 'E/A', '5D', '1M', 'YTD'] #, 'ROE'] #,'ROIC']
    if st.session_state['compare']:
        cols += ['ΔPE', 'ΔCap']
    # df_s = file_.drop(['EV', 'Competitors'], axis = 1).style
    df_s = file_[cols].style

    if not st.session_state['F']:
        df_s = df_s.apply(lambda x: list(repeat('background-color: #31668a' 
                                                if x.name in st.session_state['favs'] 
                                                else '', len(x))), 
                                 axis = 1)

    for var in ['ROA', 'Op_margin', 'rev_g', 'cash']:
        df_s = back_gradient(df_s, file_, var)
    df_s = df_s.apply(color_margin, axis = None)
    
    df_s.format({'Cap': '{:.0f}'})
    df_s.format('{:.1f}', subset = ['PEt', 'PEf', 'PB', 'PS', 'PCF', 'EV/rev', 'EV/EBITDA'])
    df_s.format('{:.2f}', subset = ['cash'])
    df_s.format('{:.3f}', subset = ['margin', 'Op_margin', 'ROA', 'ROIC', 
                                    'rev_g', 'div', 'ROE', 'E/A', '5D', '1M', 'YTD'])
    if st.session_state['compare']:
        df_s.format('{:.2f}', subset = ['ΔPE', 'ΔCap'])
    
    
    return df_s

def back_gradient(df_s, file_, var):
    df_s= df_s.background_gradient(cmap= palette, 
                                            subset = var, 
                                            low = 0, high = 0,
                                            vmin = 0.04, vmax = 0.96,
                                            gmap = file_[var].rank(pct = True))
    return df_s
    

if 'folder' not in st.session_state:
    load_folder()
else:
    # file, IS, BS = load_data()
    st.session_state['favs'] = load_favs()

    palette = diverging_palette(15, 150, as_cmap = True)
    # format_ = {'Cap' : '{:.0f' }

    col1, col2 = st.columns([2, 1])
    
    with st.sidebar:
        filtering()

    with col1:
    
        file_, IS_, BS_ = filter_file(file)
        
        cols11, cols12 = st.columns(2)
        with cols11:
            ticker = st.multiselect('', options = file_.apply(lambda x: str(x.name) + 
                                                              ' - ' + str(x['Name']), axis = 1),
                                  placeholder = 'Find ticker, company name')
        file_ = filter_by_name(file_, ticker)
        df_s = format_df(file_)
    
        # st.dataframe(df_s)
        st.dataframe(df_s, on_select = filter_by_row, 
                      selection_mode = 'single-row', key = 'file',
                      hide_index = True, height = 500) #,
                      #column_config = {'hist_rev': st.column_config.BarChartColumn()})
        st.write(file_.shape[0], ' elements')