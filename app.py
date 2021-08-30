import streamlit as st
import base64
import os

def cs_body():

    st.markdown('---')
    st.header('cheat sheet ✨')
    st.markdown('')

    # Magic commands
    st.markdown('- [`daniellewisDL/streamlit-cheat-sheet`](https://github.com/daniellewisDL/streamlit-cheat-sheet)')
    st.markdown('__How to install and import__')

    st.code('$ pip install streamlit')

    st.markdown('Import convention')
    st.code('>>> import streamlit as st')


    st.markdown('__Add widgets to sidebar__')
    st.code('''
st.sidebar.<widget>
>>> a = st.sidebar.radio(\'R:\',[1,2])
    ''')

    st.markdown('__Command line__')
    st.code('''
$ streamlit --help
$ streamlit run your_script.py
$ streamlit hello
$ streamlit config show
$ streamlit cache clear
$ streamlit docs
$ streamlit --version
    ''')

    st.markdown('__Pre-release features__')
    st.markdown('[Beta and experimental features](https://docs.streamlit.io/en/0.86.0/api.html#beta-and-experimental-features)')
    st.code('''
pip uninstall streamlit
pip install streamlit-nightly --upgrade
    ''')

    st.markdown('''
<small>Summary of the [docs](https://docs.streamlit.io/en/stable/api.html), as of [Streamlit v0.86.0](https://www.streamlit.io/).</small>
    ''', unsafe_allow_html=True)


    # ------------------------

    col1, col2 = st.columns(2)

    col1.subheader('Magic commands')
    col1.code('''# Magic commands implicitly `st.write()`
\'\'\' _This_ is some __Markdown__ \'\'\'
a=3
'dataframe:', data
    ''')

    # Display text

    col1.subheader('Display text')
    col1.code('''
st.text('Fixed width text')
st.markdown('_Markdown_') # see *
st.latex(r\'\'\' e^{i\pi} + 1 = 0 \'\'\')
st.write('Most objects') # df, err, func, keras!
st.write(['st', 'is <', 3]) # see *
st.title('My title')
st.header('My header')
st.subheader('My sub')
st.code('for i in range(8): foo()')
* optional kwarg unsafe_allow_html = True
st.caption('This is a small text')
    ''')

    # Display data

    col1.subheader('Display data')
    col1.code('''
st.dataframe(my_dataframe)
st.table(data.iloc[0:10])
st.json({'foo':'bar','fu':'ba'})
    ''')

    # Display charts

    col1.subheader('Display charts')
    col1.code('''
st.line_chart(data)
st.area_chart(data)
st.bar_chart(data)
st.pyplot(fig)
st.altair_chart(data)
st.vega_lite_chart(data)
st.plotly_chart(data)
st.bokeh_chart(data)
st.pydeck_chart(data)
st.deck_gl_chart(data)
st.graphviz_chart(data)
st.map(data)
    ''')

    # Display media

    col1.subheader('Display media')
    col1.code('''
st.image('./header.png')
st.audio(data)
st.video(data)
    ''')

    # Display interactive widgets

    col2.subheader('Display interactive widgets')
    col2.code('''
st.button('Hit me')
st.checkbox('Check me out')
st.radio('Radio', [1,2,3])
st.selectbox('Select', [1,2,3])
st.multiselect('Multiselect', [1,2,3])
st.slider('Slide me', min_value=0, max_value=10)
st.select_slider('Slide to select', options=[1,'2'])
st.text_input('Enter some text')
st.number_input('Enter a number')
st.text_area('Area for textual entry')
st.date_input('Date input')
st.time_input('Time entry')
st.file_uploader('File uploader')
st.color_picker('Pick a color')
    ''')
    col2.write('Use widgets\' returned values in variables:')
    col2.code('''
>>> for i in range(int(st.number_input('Num:'))): foo()
>>> if st.sidebar.selectbox('I:',['f']) == 'f': b()
>>> my_slider_val = st.slider('Quinn Mallory', 1, 88)
>>> st.write(slider_val)
    ''')
    col2.write('Batch widgets together in a form:')
    col2.code('''
>>> with st.form(key='my_form'):
>>> 	text_input = st.text_input(label='Enter some text')
>>> 	submit_button = st.form_submit_button(label='Submit')
    ''')

    # Control flow

    col2.subheader('Control flow')
    col2.code('''
st.stop()
    ''')

    # Lay out your app

    col2.subheader('Lay out your app')
    col2.code('''
st.container()
st.columns(spec)
>>> col1, col2 = st.columns(2)
>>> col1.subheader('Columnisation')
st.expander('Expander')
>>> with st.expander('Expand'):
>>>     st.write('Juicy deets')
    ''')


    # Display code

    col2.subheader('Display code')
    col2.code('''
st.echo()
>>> with st.echo():
>>>     st.write('Code will be executed and printed')
    ''')

    # Display progress and status

    col1.subheader('Display progress and status')
    col1.code('''
st.progress(progress_variable_1_to_100)
st.spinner()
>>> with st.spinner(text='In progress'):
>>>     time.sleep(5)
>>>     st.success('Done')
st.balloons()
st.error('Error message')
st.warning('Warning message')
st.info('Info message')
st.success('Success message')
st.exception(e)
    ''')

    # Placeholders, help, and options

    col2.subheader('Placeholders, help, and options')
    col2.code('''
st.empty()
>>> my_placeholder = st.empty()
>>> my_placeholder.text('Replaced!')
st.help(pandas.DataFrame)
st.get_option(key)
st.set_option(key, value)
st.set_page_config(layout='wide')
    ''')

    # Mutate data

    col1.subheader('Mutate data')
    col1.code('''
DeltaGenerator.add_rows(data)
>>> my_table = st.table(df1)
>>> my_table.add_rows(df2)
>>> my_chart = st.line_chart(df1)
>>> my_chart.add_rows(df2)
    ''')

    # Optimize performance

    col2.subheader('Optimize performance')
    col2.code('''
@st.cache
>>> @st.cache
... def foo(bar):
...     # Mutate bar
...     return data
>>> # Executes d1 as first time
>>> d1 = foo(ref1)
>>> # Does not execute d1; returns cached value, d1==d2
>>> d2 = foo(ref1)
>>> # Different arg, so function d1 executes
>>> d3 = foo(ref2)
    ''')

    # Store data across reruns
    col1.subheader('Store data across reruns')
    col1.code('''
st.title('Counter Example')
if 'count' not in st.session_state:
    st.session_state.count = 0
increment = st.button('Increment')
if increment:
    st.session_state.count += 1
st.write('Count = ', st.session_state.count)
    ''')

    return None


def cs_sidebar():

    st.sidebar.button("💻 Github")

    return None

# ============================

st.set_page_config(
     page_title='Streamlit tutorial',
     #layout="wide",
     initial_sidebar_state="expanded",
)


st.title('Streamlit Tutorial')
st.markdown('')
st.info('Streamlit is an open-source python framework for building web apps for Machine Learning and Data Science. We can instantly develop web apps and deploy them easily using Streamlit. Streamlit allows you to write an app the same way you write a python code. Streamlit makes it seamless to work on the interactive loop of coding and viewing results in the web app.')

st.header('Streamlit Gallery 🖼️')

with st.expander('Example 1'):
    st.markdown('''
## 💸 Stock Price Dashboard ✨

```
pip install yfinance fbprophet plotly
```
    ''')
    st.video('./asset/finance.mp4')

with st.expander('Example 2'):
    st.markdown('''
## 🙃 Cartoon StyleGAN ✨

- [`happy-jihye/Cartoon-StyleGAN`](https://github.com/happy-jihye/Cartoon-StyleGAN)

```
pip install bokeh ftfy regex tqdm gdown

# for styleclip
pip install git+https://github.com/openai/CLIP.git
```
    ''')

    st.video('./asset/cartoon-stylegan-1.mp4')
with st.expander('Example 3'):
    st.markdown('''
## 🖼️ VQGAN-CLIP ✨


```
# install python packages
pip install ftfy regex tqdm omegaconf pytorch-lightning IPython kornia imageio imageio-ffmpeg einops torch_optimizer

# clone other repositories
git clone 'https://github.com/openai/CLIP'
git clone 'https://github.com/CompVis/taming-transformers'

# download checkpoints
mkdir checkpoints
curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
```
    ''')

    st.video('./asset/vqgan.mp4')

st.markdown('---')
st.header('Streamlit API reference')
st.markdown('')
st.markdown('''
**📒 Useful resource**
- [`streamlit.io`](https://docs.streamlit.io/)
- [`awesome-streamlit`](https://github.com/MarcSkovMadsen/awesome-streamlit)
- [`streamlit gallery`](https://streamlit.io/gallery)
- [`Python Streamlit 사용법 - 프로토타입 만들기`](https://zzsza.github.io/mlops/2021/02/07/python-streamlit-dashboard/)

''')


with st.expander('Write'):
    # st.title('title')
    # st.header('header')
    # st.subheader('subheader')
    # st.write('write')

    st.markdown('''
    # title
    ## header
    ### subheader
    write
    ''')
    st.code('''
st.title('title')
st.header('header')
st.subheader('subheader')
st.write('write')
''')

with st.expander('Widget'):
    st.button('button')
    st.checkbox('checkbox')
    st.slider('slider', min_value=0, max_value=10, value=3, step=1)
    select = st.selectbox('selectbox', ['a', 'b', 'c'])
    st.write(f'select result: {select}')
    multiselect= st.multiselect('multiselect', ['a', 'b', 'c', 'd'])
    st.write(f'multiselect result: {multiselect}')

    st.code('''
st.button('button')
st.checkbox('checkbox')
st.slider('slider', min_value=0, max_value=10, value=3, step=1)
select = st.selectbox('selectbox', ['a', 'b', 'c'])
st.write(f'select result: {select}')
multiselect= st.multiselect('multiselect', ['a', 'b', 'c', 'd'])
st.write(f'multiselect result: {multiselect}')
''')

with st.expander('Input Data'):
    st.code('''
st.text_input(value)
st.text_input(label, value, type="password")
st.number_input(label, value)
st.text_area(label, value)
st.date_input(label, value)
st.time_input(label, value)
''')

with st.expander('Message'):
    st.info('info')
    st.error('error')
    st.warning('warning')
    st.success('success')
    st.code('''
st.info('info')
st.error('error')
st.warning('warning')
st.success('success')
    ''')
cs_body()
