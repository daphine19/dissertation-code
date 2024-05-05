import streamlit as st
import plotly as plotly
import pandas as pd 
import plotly.graph_objects as go
import numpy as np

#variables 
sitting_df = None
standing_df = None 
load_df = None 
csv_df = None
posture_df = None
time_interval_minutes = (1 / 15 )/60 #calculated from 15Hz and then divided by 60 to get the minutes. 
post_time_above =None
load_time_above = None

#Page layout 
st.set_page_config(page_title="Posture Dashboord", layout = "wide")
st.header("Posture Dashboard")
st.markdown('<style> div.block-container{padding-top:20px;}</style>', unsafe_allow_html=True)
st.warning("To get started, please upload the file and the page will refresh automatically. Read how to guides for any trouble shooting ", icon="🚨")
with open('styles.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

#Functions 
def visualize_behaviour_data(df, title):
    global hourly_data, time_interval_minutes

    hourly_data = pd.DataFrame() 
    fig_line = None

    if df is not None:
        df['Timestamp'] = pd.date_range(start='8:00:00', periods=len(df), freq='66.6667ms')
        df.set_index('Timestamp', inplace=True)

        df = df.apply(pd.to_numeric, errors='ignore')
        numeric_columns = df.select_dtypes(include=np.number)

        hourly_data = numeric_columns.resample('10min').mean()
        column = df.columns[0]
    
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=hourly_data.index, y=hourly_data[column], mode='lines+markers'))
        fig_line.update_layout(title=title, xaxis_title='Time (Hour)', yaxis_title='ROM')


        tick_values = pd.date_range(start=hourly_data.index.min(), end=hourly_data.index.max(), freq='1H')
        fig_line.update_xaxes(tickmode='array', tickvals=tick_values, tickformat='%H:%M')

        fig_line.update_yaxes(nticks=20) 
        st.plotly_chart(fig_line, use_container_width=True)

    return hourly_data, fig_line

#Area chart for time distibution 
def visualise_above_range(area_df, title2):
    global df_max_value, colors

    fig_area = None
    if area_df is not None:
        area_df['Timestamp'] = pd.date_range(start='8:00:00', periods=len(area_df), freq='66.6667ms')
        area_df.set_index('Timestamp', inplace=True)

        area_df = area_df.apply(pd.to_numeric, errors='ignore')
        numeric_columns = area_df.select_dtypes(include=np.number)

        hourly_data_2 = numeric_columns.resample('1min').mean()

        area_column = area_df.columns[0]

        hourly_data_2['val'] = hourly_data_2[area_column]

        df_max_value =  area_df.iloc[:, 0].max()

        bins =[50,60,70,80,90,df_max_value]

        hourly_data_2['range'] = pd.cut(hourly_data_2['val'], bins=bins, labels=['50-59','60-69', '70-79', '80-89', '>90'])

        fig_area = go.Figure()
        colors = {
            '50-59': '#90EE90',
            '60-69': '#008000',
            '70-79': '#FFA500',
            '80-89': '#FF0000',
            '>90': '#8B0000'}
        
        for label, group in hourly_data_2.groupby('range'):
            fig_area.add_trace(go.Scatter(x=group.index, y=group['val'], stackgroup='one', name=label, line=dict(color=colors.get(label,'#D3D3D3'))))
        fig_area.update_layout(title= title2, xaxis_title='Time (Hour)', yaxis_title='ROM')
        fig_area.update_xaxes(tickformat='%H:%M')
        fig_area.update_yaxes(range = [20, df_max_value],nticks=20)
        st.plotly_chart(fig_area, use_container_width=True)

    return fig_area

#Pie chart for total time:
def total_time_range(pie_df):
    fig_donut = None
    if pie_df is not None:
        df_bins =[50,60,70,80,90,df_max_value]
        pie_df['threshold'] = pd.cut(pie_df[pie_df.columns[0]], bins=df_bins, labels=['50-59','60-69', '70-79', '80-89', '>90'])

        time_spent =pie_df.groupby('threshold').size()*time_interval_minutes

        labels =list(time_spent.index)
        values =[round(value, 2) for value in time_spent.values]
                            
        fig_donut = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                textinfo='label+value',
                hole=0.4,
                marker=dict(colors=[colors.get(label, '#D3D3D3') for label in labels])
    )
])      
        fig_donut.update_layout(title='Time Spent in Each Bin Range (Minutes)', showlegend=True)
        st.plotly_chart(fig_donut, use_container_width=True)

    return fig_donut 

# Plot posture behavior over time
def plot_behavior(behaviour_df):
    if behaviour_df is not None:   
        behaviour_df['Timestamp'] = pd.date_range(start='8:00:00', periods=len(behaviour_df), freq='66.6667ms')
        behaviour_df.set_index('Timestamp', inplace=True)
        
        hourly_data = behaviour_df.resample('10min').mean()
        column2 = behaviour_df.columns[0]

        start_time = hourly_data.index.min()
        hourly_data['time_intervals'] = ((hourly_data.index - start_time).total_seconds() / 60).astype(int)
        hourly_data['time_intervals'] = (hourly_data['time_intervals'] // 10) * 10

        if not behaviour_df.empty:
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=hourly_data['time_intervals'], y=hourly_data[column2], mode='lines+markers'))
            fig_line.update_layout(title="Standing behaviour over day",xaxis_title='Time Samples', yaxis_title='ROM')   

            x_axis_min = hourly_data['time_intervals'].min()
            x_axis_max = hourly_data['time_intervals'].max()
            fig_line.update_xaxes(tickmode='array', tickvals=list(range(x_axis_min, x_axis_max+1, 20)))

            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning("No data available for standing behavior")

#Main app
with st.sidebar:
    st.subheader("Navigation and Upload ")
    with st.expander("Upload file"):
        patientName = st.text_input('Please enter patients name') 
        durationRecorded = st.text_input('How long was the data collected for (Hours)')
        painIntensity = st.slider('Pain Intensity', min_value=1, max_value=5, step=1)
        uploadedFile = st.file_uploader('Upload your Excel file', accept_multiple_files=False, type=['xlsx','xls'])

        if uploadedFile is not None:
            with st.spinner("Generating Dashboard..."):
                try:
                    if uploadedFile.name.endswith('.csv'):
                            csv_df = pd.read_csv(uploadedFile)
                    elif uploadedFile.name.endswith('.xlsx') or uploadedFile.name.endswith('.xls'):
                        xls = pd.ExcelFile(uploadedFile)
                        posture_df = pd.DataFrame()
                        load_df = pd.DataFrame()
                        sitting_df = pd.DataFrame()
                        standing_df = pd.DataFrame()
                        for i, sheet_name in enumerate(xls.sheet_names):
                            if i == 0:
                                posture_df = pd.read_excel(xls, sheet_name=sheet_name)
                            elif i == 1:
                                load_df = pd.read_excel(xls, sheet_name=sheet_name)
                            elif i == 2:
                                sitting_df = pd.read_excel(xls, sheet_name=sheet_name)
                            elif i == 3:
                                standing_df = pd.read_excel(xls, sheet_name= sheet_name)
                    else:
                        st.write("Unsupported file format. Please upload an Excel file")
                except Exception as e:
                    st.write(f"An error occured: {e}")
                else:
                    st.write("Please upload file to run application")

    with st.expander("Patient Notes"):
        notes = st.text_area("Record pain points and any additional recommendations discussed")
        if st.button("Save Notes"):
            st.success("Notes saved successfully!")
    
    with st.expander("How to guides"):
        st.write("Upload excel sheet with percentage values as flexion.")
        st.divider()
        st.write("Ensure the sheets in the excel file are uploaded in the order of posture, load, sit then stand.")
        st.divider()
        st.write("To print the page with all visuals and notes, follow these steps:")
        st.write("- Select the options bar icon from the top right of the screen.")
        st.write("- Then select 'Print'.")
        st.write("To download, follow above steps then:")
        st.write("- Select 'More options'.")
        st.write("- Choose 'Open in PDF preview'.")
        st.write("- Save the file.")
        st.write("Close the navigation window for better results.")   
        st.divider()
        
    with st.expander("Thresholds explained"):
        st.write("<span style='color:green'>50-60% - Good range but could be better.</span>", unsafe_allow_html=True)
        st.write("<span style='color:orange'>60-70% - Good but could be bad based on time spent in position.</span>", unsafe_allow_html=True)
        st.write("<span style='color:red'>70-80% - Bad range.</span>", unsafe_allow_html=True)
        st.write("<span style='color:#8B0000'>80-100% - Very dangerous, should be avoided.</span>", unsafe_allow_html=True)

#Calculating total time spent above 50%                      
if posture_df is not None and load_df is not None:
    if not posture_df.empty and not load_df.empty:
        if posture_df.columns[0] in posture_df.columns and load_df.columns[0] in load_df.columns:
            post_time_above = round(len(posture_df[posture_df.iloc[:, 0] > 50]) * time_interval_minutes, 2)
            load_time_above = round(len(load_df[load_df.iloc[:, 0] > 50]) * time_interval_minutes, 2)
        else:
            st.warning("Dataframes 'posture_df' or 'load_df' are missing required columns.")
    else:
        st.warning("Dataframes 'posture_df' or 'load_df' are empty. Cannot calculate time spent above 50%.")

#Identifying data
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write("**Patient Name**")
        st.caption(patientName)
      
    with col2:
        st.write("**Duration Recorded**")
        st.caption(f'{durationRecorded} hours')

    with col3:
        st.write("**Back Pain Intensity (0-5)**")
        st.caption(painIntensity)

    with col4:
        st.write("**Time Spent Above 50% Posture**")
        st.caption(f'{post_time_above} minutes')

    with col5:
        st.write("**Time Spent Above 50% Load**")
        st.caption(f'{load_time_above} minutes')

    st.write("---")

#plot variable charts through day 
with st.container():
    st.subheader("Variable Distribution through Day")
    left, right = st.columns(2)
    with left:
        fig_post_line =visualize_behaviour_data(posture_df,'Posture Distribution Over Time (10 mins increment)')
        fig_load_line = post_hourly_data = hourly_data

    with right:
        visualize_behaviour_data(load_df,'Load Distribution Over Time (10 mins increment)')
        load_hourly_data = hourly_data

#plot above range charts 
with st.container():
    st.subheader('Posture above 50%')
    col1, col2 = st.columns([3,2])
    with col1:
        fig_post_area =visualise_above_range(posture_df,'Posture Distribution above 50%')
    with col2:
        fig_post_donut = total_time_range(posture_df)

with st.container():
    st.subheader('Load above 50%')
    col1, col2 = st.columns([3,2])
    with col1:
        fig_load_area = visualise_above_range(load_df,'Load_distribution above 50%')
    with col2:
        fig_load_donut = total_time_range(load_df)

#Create charts for sit and stand
with st.container():
    st.subheader("Standing and Sitting Distibution")

    col1, col2, col3 = st.columns(3)
    with col1:
        fig_stand_line =plot_behavior(standing_df)

    with col2:
        fig_sit_line = plot_behavior(sitting_df)

    #Plot pie chart of time in each position using thresholds
    with col3:
        if posture_df is not None and not posture_df.empty:
            full_post_time = len(posture_df) * time_interval_minutes
            if standing_df is not None and not standing_df.empty:
                standing_time = round(len(standing_df) * time_interval_minutes, 2)
            else:
                standing_time = 0
            if sitting_df is not None and not sitting_df.empty:
                sitting_time = round(len(sitting_df) * time_interval_minutes, 2)
            else:
                sitting_time = 0

            other_activities = round(full_post_time - (sitting_time + standing_time), 2)

            labels = [f'Standing\n({standing_time})', f'Sitting\n({sitting_time})', f'Other-Activities\n({other_activities})']
            values = [standing_time, sitting_time, other_activities]
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+value')])
            fig_pie.update_layout(title='Time Spent in Posture Positions (Minutes)', showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        #else:
                #st.warning("Posture DataFrame is missing or empty. Cannot visualize time spent in posture positions.")

#Plot posture and load vs time 
with st.container():
    st.subheader ("Load, Posture and Time Relationship")
    col1, col2 =st.columns(2)
    with col1:
        if posture_df is not None and load_df is not None:
            post_values = post_hourly_data.iloc[:,0].values
            load_values = load_hourly_data.iloc[:,0].values

            load_color = '#0047AB'
            post_color= '#097969'

            fig_line2 = go.Figure()
            fig_line2.add_trace(go.Scatter(x=post_hourly_data.index, y=post_values, mode='lines+markers', name='Posture',line=dict(color=post_color)))
            fig_line2.add_trace(go.Scatter(x=load_hourly_data.index, y=load_values, mode='lines+markers', name='Load',line=dict(color=load_color)))
            fig_line2.update_layout(title='Posture and Load Distribution Over Time', xaxis_title='Time (Hour)', yaxis_title='ROM')
            fig_line2.update_yaxes(nticks=20) 
            fig_line2.update_xaxes(tickformat='%H:%M')
            st.plotly_chart(fig_line2, use_container_width=True)

    with col2:
        if posture_df is not None and load_df is not None:
            post_col = posture_df.columns[0]
            load_col = load_df.columns[0]

            load_df[load_col] = pd.to_numeric(load_df[load_col], errors='coerce')
            posture_df[post_col] = pd.to_numeric(posture_df[post_col], errors='coerce')

            # Extract the maximum value from the DataFrame column and convert it to an integer
            load_max_value = int(load_df[load_col].max())
            post_max_value = int(posture_df[post_col].max())

            # Define bin ranges for load and posture
            load_bins = [(60, 70), (70, 80), (80, 90), (90, load_max_value)]  
            post_bins = [(60, 70), (70, 80), (80, 90), (90, post_max_value)] 

            time_spent = {}

            for load_range in load_bins:
                for post_range in post_bins:
                    # Filter data where load and posture are within the specified bin ranges
                    filtered_data = ((posture_df[post_col] >= post_range[0]) & (posture_df[post_col] < post_range[1])) & \
                                    ((load_df[load_col] >= load_range[0]) & (load_df[load_col] < load_range[1]))
                    
                    total_time = (filtered_data.sum() * time_interval_minutes)
                    time_spent[f'Load {load_range[0]}-{load_range[1]} and Posture {post_range[0]}-{post_range[1]}'] = total_time

            labels = list(time_spent.keys())
            values = list(time_spent.values())

            fig_bar = go.Figure()

            for label, value in time_spent.items():
                fig_bar.add_trace(go.Bar(x=[label], y=[value], name=label))
                fig_bar.update_layout(title='Total Time Spent in Bin Range Combinations', xaxis_title='Bin Range Combination', yaxis_title='Total Time (Minutes)', showlegend=False, yaxis=dict(tickmode='linear', dtick=1))
            st.plotly_chart(fig_bar, use_container_width=True)

#saved notes            
st.divider()
st.subheader('Discussed Notes')
with st.container(border=True):
    st.write(notes) 

#footer   
st.markdown("""
    <hr style='border: 1px solid #0096FF; width: 100%;'>
    <p style='text-align: center;'>To print dashboard, check how to guide and for further support, please seek assistance from your assessor.</p>
""", unsafe_allow_html=True)

# hide streamlit styles 
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

    



