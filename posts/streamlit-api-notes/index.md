---
categories:
- python
- streamlit
- numpy
- pandas
- notes
date: 2022-1-2
description: My notes and reference examples for working with the Streamlit API.
hide: false
layout: post
search_exclude: false
title: Notes on the Streamlit API
toc: false

aliases:
- /Notes-on-Streamlit-API/
---



* [Overview](#overview)
* [Streamlit](#streamlit)
* [Magic Commands](#magic-commands)
* [Display Text](#display-text)
* [Display Data](#display-data)
* [Display Charts](#display-charts)
* [Display Media](#display-media)
* [Add Widgets to Sidebar](#add-widgets-to-sidebar)
* [Columns](#columns)
* [Control Flow](#control-flow)
* [Display Interactive Widgets](#display-interactive-widgets)
* [Command Line](#command-line)
* [Mutate Data](#mutate-data)
* [Display and Execute Code](#display-and-execute-code)
* [Placeholders, help, and options](#placeholders,-help,-and-options)
* [Page Configuration](#page-configuration)
* [Cache Data Objects](#cache-data-objects)
* [Cache Non-data Objects](#cache-non-data-objects)
* [Display Progress and Status](#display-progress-and-status)



## Overview

Here are some notes and reference code for working with the [Streamlit API](https://docs.streamlit.io/library/api-reference).


## Streamlit
* [Streamlit - The fastest way to build and share data apps](https://streamlit.io/)

- Turns data scripts into shareable web apps
- `pip install streamlit`
- Test Installation: `streamlit hello`
- Run apps: `streamlit run main.py`
- Format text using [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)



## Magic Commands

**replit:** [https://replit.com/@innominate817/streamlit-magic-commands#main.py](https://replit.com/@innominate817/streamlit-magic-commands#main.py)

```python
import streamlit as st
import pandas as pd

d = {'col1': [1,2], 'col2': [3,4]}
data = pd.DataFrame(data=d)

# Magic commands implicitly call st.write()
'_This_ is some **Markdown***'
num = 3
'dataframe:', data
```



## Display Text

**replit:** [https://replit.com/@innominate817/streamlit-display-text#main.py](https://replit.com/@innominate817/streamlit-display-text#main.py)

```python
import streamlit as st

st.text('Fixed width text')
st.markdown('_Markdown_') # see *
st.caption('Balloons. Hundreds of them...')
st.latex(r''' e^{i\pi} + 1 = 0 ''')
st.write('Most objects') # df, err, func, keras!
st.write(['st', 'is <', 3]) # see *
st.title('My title')
st.header('My header')
st.subheader('My sub')
st.code('for i in range(8): foo()')
```



## Display Data

**replit:** [https://replit.com/@innominate817/streamlit-display-data](https://replit.com/@innominate817/streamlit-display-data)

```python
import streamlit as st
import pandas as pd

d = [{
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4
}, {
    'a': 100,
    'b': 200,
    'c': 300,
    'd': 400
}, {
    'a': 1000,
    'b': 2000,
    'c': 3000,
    'd': 4000
}]
data = pd.DataFrame(data=d)

st.dataframe(data)
st.table(data.iloc[0:2])
st.json({'foo': 'bar', 'fu': 'ba'})
st.metric(label="Temp", value="273 K", delta="1.2 K")

```





## Display Charts

**replit #1:** [https://replit.com/@innominate817/streamlit-display-charts#main.py](https://replit.com/@innominate817/streamlit-display-charts#main.py)

```python
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import altair as alt
import plotly.figure_factory as ff
from bokeh.plotting import figure

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])

st.line_chart(chart_data)

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])

st.area_chart(chart_data)

chart_data = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])

st.bar_chart(chart_data)

arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.pyplot(fig)

df = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])

c = alt.Chart(df).mark_circle().encode(x='a',
                                       y='b',
                                       size='c',
                                       color='c',
                                       tooltip=['a', 'b', 'c'])

st.altair_chart(c, use_container_width=True)

df = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])

st.vega_lite_chart(
    df, {
        'mark': {
            'type': 'circle',
            'tooltip': True
        },
        'encoding': {
            'x': {
                'field': 'a',
                'type': 'quantitative'
            },
            'y': {
                'field': 'b',
                'type': 'quantitative'
            },
            'size': {
                'field': 'c',
                'type': 'quantitative'
            },
            'color': {
                'field': 'c',
                'type': 'quantitative'
            },
        },
    })

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

# Group data together
hist_data = [x1, x2, x3]

group_labels = ['Group 1', 'Group 2', 'Group 3']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])

# Plot!
st.plotly_chart(fig, use_container_width=True)

x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

p = figure(title='simple line example', x_axis_label='x', y_axis_label='y')

p.line(x, y, legend_label='Trend', line_width=2)

st.bokeh_chart(p, use_container_width=True)

```



**replit #2:** [https://replit.com/@innominate817/streamlit-display-charts-2#main.py](https://replit.com/@innominate817/streamlit-display-charts-2#main.py)

```python
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import graphviz as graphviz

df = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
                  columns=['lat', 'lon'])

st.pydeck_chart(
    pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=37.76,
            longitude=-122.4,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=df,
                get_position='[lon, lat]',
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=df,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
            ),
        ],
    ))

# Create a graphlib graph object
graph = graphviz.Digraph()
graph.edge('run', 'intr')
graph.edge('intr', 'runbl')
graph.edge('runbl', 'run')
graph.edge('run', 'kernel')
graph.edge('kernel', 'zombie')
graph.edge('kernel', 'sleep')
graph.edge('kernel', 'runmem')
graph.edge('sleep', 'swap')
graph.edge('swap', 'runswap')
graph.edge('runswap', 'new')
graph.edge('runswap', 'runmem')
graph.edge('new', 'runmem')
graph.edge('sleep', 'runmem')

st.graphviz_chart(graph)

df = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
                  columns=['lat', 'lon'])
st.map(df)

```







## Display Media

**replit:** [https://replit.com/@innominate817/streamlit-display-media#main.py](https://replit.com/@innominate817/streamlit-display-media#main.py)

**Note:** audio does not seem to pass through in replit

```python
import streamlit as st
from PIL import Image

image = Image.open('sunrise.jpg')
st.image(image, caption='Sunrise by the mountains')

audio_file = open('audio_sample.mp3', 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/mp3')

video_file = open('video_sample.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

```





## Add Widgets to Sidebar

**replit:** [https://replit.com/@innominate817/streamlit-add-widgets-to-sidebar#main.py](https://replit.com/@innominate817/streamlit-add-widgets-to-sidebar#main.py) 

```python
import streamlit as st

# Add individual widget to side bar
radio1 = st.sidebar.radio('Option #1:', [1, 2])

# Add group of widgets to sidebar
with st.sidebar:
    radio2 = st.radio('Option #2:', [3, 4])
    checkbox1 = st.checkbox('Check me')

if radio2 == 3:
    st.write('You selected number 3.')
else:
    st.write("You selected number 4.")

if checkbox1:
    st.write("checkbox checked")
else:
    st.write("checkbox unchecked")

```





## Columns

**replit:** [https://replit.com/@innominate817/streamlit-columns#main.py](https://replit.com/@innominate817/streamlit-columns#main.py)

```python
import streamlit as st

col1, col2, col3 = st.columns(3)

with col1:
    st.header("A cat")
    st.image("https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg?cs=srgb&dl=pexels-evg-culture-1170986.jpg&fm=jpg&w=640&h=960")

with col2:
    st.header("A dog")
    st.image("https://images.pexels.com/photos/2252311/pexels-photo-2252311.jpeg?cs=srgb&dl=pexels-laura-stanley-2252311.jpg&fm=jpg&w=640&h=959")

with col3:
    st.header("An owl")
    st.image("https://images.pexels.com/photos/5883285/pexels-photo-5883285.jpeg?cs=srgb&dl=pexels-mehmet-turgut-kirkgoz-5883285.jpg&fm=jpg&w=640&h=960")
```





## Control Flow

**replit:** [https://replit.com/@innominate817/streamlit-control-flow#main.py](https://replit.com/@innominate817/streamlit-control-flow#main.py) 

```python
import streamlit as st

name = st.text_input('Name')
if not name:
    st.warning('Please input a name.')
    st.stop()
st.success('Thank you for inputting a name.')

with st.form("my_form"):
    st.write("Inside the form")
    slider_val = st.slider("Form slider")
    checkbox_val = st.checkbox("Form checkbox")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("slider", slider_val, "checkbox", checkbox_val)

st.write("Outside the form")

```







## Display Interactive Widgets

**replit:** [https://replit.com/@innominate817/streamlit-interactive-widgets#main.py](https://replit.com/@innominate817/streamlit-interactive-widgets#main.py)

```python
import streamlit as st
import numpy as np
import pandas as pd
import io

st.header("Button")
if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')

st.header("Download Buttons")
df = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',
)

text_contents = '''This is some text'''
st.download_button('Download some text', text_contents)

binary_contents = b'example content'
# Defaults to 'application/octet-stream'
st.download_button('Download binary file', binary_contents)

with open("flower.jpg", "rb") as file:
    btn = st.download_button(label="Download image",
                             data=file,
                             file_name="flower.png",
                             mime="image/png")

st.header("Checkbox")
agree = st.checkbox('I agree')

if agree:
    st.write('Great!')

st.header("Radio Button")
genre = st.radio("What's your favorite movie genre",
                 ('Comedy', 'Drama', 'Documentary'))

if genre == 'Comedy':
    st.write('You selected comedy.')
else:
    st.write("You didn't select comedy.")

st.header("Selectbox")
option = st.selectbox('How would you like to be contacted?',
                      ('Email', 'Home phone', 'Mobile phone'))

st.write('You selected:', option)

st.header("Multiselect")
options_multi = st.multiselect('What are your favorite colors',
                               ['Green', 'Yellow', 'Red', 'Blue'],
                               ['Yellow', 'Red'])

st.write('You selected:', options_multi)

st.header("Sliders")
st.subheader('Basic')
st.slider('Slide me', min_value=0, max_value=10)
st.subheader('Range Slider')
values = st.slider('Select a range of values', 0.0, 100.0, (25.0, 75.0))
st.write('Values:', values)

st.subheader('Range Time Slider')
from datetime import time

appointment = st.slider("Schedule your appointment:",
                        value=(time(11, 30), time(12, 45)))
st.write("You're scheduled for:", appointment)

st.subheader('Datetime Slider')
from datetime import datetime

start_time = st.slider("When do you start?",
                       value=datetime(2020, 1, 1, 9, 30),
                       format="MM/DD/YY - hh:mm")
st.write("Start time:", start_time)

st.header("Select Sliders")
color = st.select_slider(
    'Select a color of the rainbow',
    options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
st.write('My favorite color is', color)

start_color, end_color = st.select_slider(
    'Select a range of color wavelength',
    options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
    value=('red', 'blue'))
st.write('You selected wavelengths between', start_color, 'and', end_color)

st.header("Text Input")
title = st.text_input('Movie title', 'Life of Brian')
st.write('The current movie title is', title)

st.header("Number Input")
number = st.number_input('Insert a number')
st.write('The current number is ', number)

st.header("Text Area")
txt = st.text_area(
    'Text to analyze', '''
     It was the best of times, it was the worst of times, it was
     the age of wisdom, it was the age of foolishness, it was
     the epoch of belief, it was the epoch of incredulity, it
     was the season of Light, it was the season of Darkness, it
     was the spring of hope, it was the winter of despair, (...)
     ''')
st.write('First word:', txt.split(',')[0:2])

st.header("Date Input")
from datetime import date

d = st.date_input("When's your birthday", date(2019, 7, 6))
st.write('Your birthday is:', d)

st.header("Time Input")
t = st.time_input('Set an alarm for', time(8, 45))
st.write('Alarm is set for', t)

st.header("File Uploader")
st.subheader("Single File")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
st.subheader("Multiple Files")
uploaded_files = st.file_uploader("Choose a CSV file",
                                  accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)

st.header("Color Picker")
color = st.color_picker('Pick A Color', '#00f900')
st.write('The current color is', color)

```





## Command Line

```bash
streamlit --help
streamlit run your_script.py
streamlit hello
streamlit config show
streamlit cache clear
streamlit docs
streamlit --version
```



## Mutate Data

**replit:** [https://replit.com/@innominate817/streamlit-mutate-data#main.py](https://replit.com/@innominate817/streamlit-mutate-data#main.py)

```python
import streamlit as st
import pandas as pd
import numpy as np

st.header("Mutate Table")
df1 = pd.DataFrame(np.random.randn(50, 20),
                   columns=('col %d' % i for i in range(20)))

my_table = st.table(df1)

df2 = pd.DataFrame(np.random.randn(50, 20),
                   columns=('col %d' % i for i in range(20)))

my_table.add_rows(df2)
# Now the table shown in the Streamlit app contains the data for
# df1 followed by the data for df2.

st.header("Mutate Chart")
# Assuming df1 and df2 from the example above still exist...
my_chart = st.line_chart(df1)
my_chart.add_rows(df2)
# Now the chart shown in the Streamlit app contains the data for
# df1 followed by the data for df2.
```





## Display and Execute Code

**replit:** [https://replit.com/@innominate817/streamlit-echo-code#main.py](https://replit.com/@innominate817/streamlit-echo-code#main.py)

```python
import streamlit as st

with st.echo():
    st.write('This code will be printed')
    num = 2 + 2
    st.write(f'Sum: {num}')
```





## Placeholders, help, and options

**replit:** [https://replit.com/@innominate817/streamlit-placeholders-help-options#main.py](https://replit.com/@innominate817/streamlit-placeholders-help-options#main.py)

```python
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout='wide')

# Replace any single element.
element = st.empty()
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
element.line_chart(chart_data)
element.text_input("New Text")  # Replaces previous.

# Insert out of order.
elements = st.container()
elements.line_chart(chart_data)
st.write("Hello")
elements.text_input("Some more new text")  # Appears above "Hello".

st.help(pd.DataFrame)
```



## Page Configuration

**replit:** [https://replit.com/@innominate817/streamlit-page-configuration#main.py](https://replit.com/@innominate817/streamlit-page-configuration#main.py)

```python
import streamlit as st

st.set_page_config(page_title="Ex-stream-ly Cool App",
                   page_icon="🧊",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={
                       'Get Help':
                       'https://www.extremelycoolapp.com/help',
                       'Report a bug':
                       "https://www.extremelycoolapp.com/bug",
                       'About':
                       "# This is a header. This is an *extremely* cool app!"
                   })

```





## Cache Data Objects

```python
import streamlit
import pandas as pd
import numpy as np

# E.g. Dataframe computation, storing downloaded data, etc.
@st.experimental_memo
def foo(bar):
    # Do something expensive and return data
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
    return chart_data
# Executes foo
d1 = foo(ref1)
# Does not execute foo
# Returns cached item by value, d1 == d2
d2 = foo(ref1)
# Different arg, so function foo executes
d3 = foo(ref2)
```



## Cache Non-data Objects

```python
import streamlit as st

# E.g. TensorFlow session, database connection, etc.
@st.experimental_singleton
def foo(bar):
    # Create and return a non-data object
    return session
# Executes foo
s1 = foo(ref1)
# Does not execute foo
# Returns cached item by reference, d1 == d2
s2 = foo(ref1)
# Different arg, so function foo executes
s3 = foo(ref2)
```



## Display Progress and Status

**replit:** [https://replit.com/@innominate817/streamlit-display-progress-and-status#main.py](https://replit.com/@innominate817/streamlit-display-progress-and-status#main.py)

```python
import streamlit as st
import time

with st.spinner('Wait for it...'):
    time.sleep(5)
st.success('Done!')

my_bar = st.progress(0)

for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1)

st.balloons()

st.error('Error message')

st.warning('Warning message')

st.info('Info message')

st.success('Success message')

e = RuntimeError('This is an exception of type RuntimeError')
st.exception(e)

```




**References:**

* [Streamlit Documentation](https://docs.streamlit.io/)



<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->
