import os, sys
import datetime, subprocess
from mega import Mega
now_dir = os.getcwd()
sys.path.append(now_dir)
import logging
import shutil
import threading
import traceback
import warnings
from random import shuffle
from subprocess import Popen
from time import sleep
import json
import pathlib
import yt_dlp
import fairseq
import faiss
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans

from configs.config import Config
from i18n.i18n import I18nAuto
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from infer.modules.uvr5.modules import uvr
from infer.modules.vc.modules import VC
logging.getLogger("numba").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


load_dotenv()
config = Config()
vc = VC(config)


def get_youtube_video_id(url, ignore_playlist=True):
    """
    Examples:
    http://youtu.be/SA2iWivDJiE
    http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    http://www.youtube.com/embed/SA2iWivDJiE
    http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        if query.path[1:] == 'watch':
            return query.query[2:]
        return query.path[1:]

    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
            # use case: get playlist id not current video in playlist
            with suppress(KeyError):
                return parse_qs(query.query)['list'][0]
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/watch/':
            return query.path.split('/')[1]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]

    # returns None for invalid YouTube url
    return None


def yt_download(link):
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': '%(title)s',
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'extractaudio': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        download_path = ydl.prepare_filename(result, outtmpl='%(title)s.mp3')

    return download_path


def raise_exception(error_msg, is_webui):
    if is_webui:
        raise gr.Error(error_msg)
    else:
        raise Exception(error_msg)


def download_from_url(url, model):
    if model =='':
        try:
            model = url.split('/')[-1]
        except:
            return "You need to name your model. For example: My-Model"
    url=url.replace('/blob/main/','/resolve/main/')
    model=model.replace('.pth','').replace('.index','').replace('.zip','')
    if url == '':
        return "URL cannot be left empty."
    url = url.strip()
    zip_dirs = ["zips", "unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs("zips", exist_ok=True)
    os.makedirs("unzips", exist_ok=True)
    zipfile = model + '.zip'
    zipfile_path = './zips/' + zipfile
    try:
        if url.endswith('.pth'):
            subprocess.run(["wget", url, "-O", f'./assets/weights/{model}.pth'])
            return f"Sucessfully downloaded as {model}.pth"
        if url.endswith('.index'):
            if not os.path.exists(f'./logs/{model}'): os.makedirs(f'./logs/{model}')
            subprocess.run(["wget", url, "-O", f'./logs/{model}/added_{model}.index'])
            return f"Successfully downloaded as added_{model}.index"
        if "drive.google.com" in url:
            subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
        elif "mega.nz" in url:
            m = Mega()
            m.download_url(url, './zips')
        else:
            subprocess.run(["wget", url, "-O", zipfile_path])
        for filename in os.listdir("./zips"):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join("./zips/",filename)
                shutil.unpack_archive(zipfile_path, "./unzips", 'zip')
            else:
                return "No zipfile found."
        for root, dirs, files in os.walk('./unzips'):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".index"):
                    os.mkdir(f'./logs/{model}')
                    shutil.copy2(file_path,f'./logs/{model}')
                elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                    shutil.copy(file_path,f'./assets/weights/{model}.pth')
        shutil.rmtree("zips")
        shutil.rmtree("unzips")
        return "Success."
    except:
        return "There's been an error."

def upload_to_dataset(files, dir):
    if dir == '':
        dir = './dataset/'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(dir):
        os.makedirs(dir)
    for file in files:
        path=file.name
        shutil.copy2(path,dir)
    try:
        gr.Info(i18n("处理数据"))
    except:
        pass
    return i18n("处理数据"), {"value":dir,"__type__":"update"}

def download_model_files(model):
    model_found = False
    index_found = False
    if os.path.exists(f'./assets/weights/{model}.pth'): model_found = True
    if os.path.exists(f'./logs/{model}'):
        for file in os.listdir(f'./logs/{model}'):
            if file.endswith('.index') and 'added' in file:
                log_file = file
                index_found = True
    if model_found and index_found:
        return [f'./assets/weights/{model}.pth', f'./logs/{model}/{log_file}'], "Done"
    elif model_found and not index_found:
        return f'./assets/weights/{model}.pth', "Could not find Index file."
    elif index_found and not model_found:
        return f'./logs/{model}/{log_file}', f'Make sure the Voice Name is correct. I could not find {model}.pth'
    else:
        return None, f'Could not find {model}.pth or corresponding Index file.'

def update_visibility(visible):
    if visible:
        return {"visible":True,"__type__":"update"},{"visible":True,"__type__":"update"}
    else:
        return {"visible":False,"__type__":"update"},{"visible":False,"__type__":"update"}

def get_pretrains(string):
    pretrains = []
    for file in os.listdir('assets/pretrained_v2'):
        if string in file:
            pretrains.append(os.path.join('assets/pretrained_v2',file))
    return pretrains


with gr.Blocks(theme=gr.themes.Soft(), title="RVC 💻") as app:
    gr.HTML("<h1> The RVC EASY WEBUI 💻 </h1>")
        
            sid = gr.Dropdown(label=i18n("推理音色"), choices=sorted(names))
          with gr.Column() as yt_link_col:
              song_input = gr.Text(label='Song input', info='Link to a song on YouTube or full path to a local file. For file upload, click the button below.')
                        

            sid.change(fn=vc.get_vc, inputs=[sid], outputs=[spk_item])
            gr.Markdown(
                value=i18n(
                    "男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. "
                )
            )
            vc_input3 = gr.Audio(label="上传音频（长度小于90秒）")
            vc_transform0 = gr.Number(
                label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0
            )
            f0method0 = gr.Radio(
                label=i18n(
                    "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU"
                ),
                choices=["crepe", "rmvpe"],
                value="rmvpe",
                interactive=True,
            )
        
            with gr.Column():
                file_index1 = gr.Textbox(
                    label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                    value="",
                    interactive=False,
                    visible=False,
                )
            file_index2 = gr.Dropdown(
                label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                choices=sorted(index_paths),
                interactive=True,
            )
            index_rate1 = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("检索特征占比"),
                value=0.88,
                interactive=True,
            )
            resample_sr0 = gr.Slider(
                minimum=0,
                maximum=48000,
                label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                value=0,
                step=1,
                interactive=True,
            )
            rms_mix_rate0 = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n(
                    "输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"
                ),
                value=1,
                interactive=True,
            )
            protect0 = gr.Slider(
                minimum=0,
                maximum=0.5,
                label=i18n(
                    "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                ),
                value=0.33,
                step=0.01,
                interactive=True,
            )
            f0_file = gr.File(
                label=i18n("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调")
            )
            but0 = gr.Button(i18n("转换"), variant="primary")
            vc_output1 = gr.Textbox(label=i18n("输出信息"))
            vc_output2 = gr.Audio(label=i18n("输出音频(右下角三个点,点了可以下载)"))
            but0.click(
                vc.vc_single,
                [
                    song_input,
                    spk_item,
                    vc_input3,
                    vc_transform0,
                    f0_file,
                    f0method0,
                    file_index1,
                    file_index2,
                    # file_big_npy1,
                    index_rate1,
                    filter_radius0,
                    resample_sr0,
                    rms_mix_rate0,
                    protect0,
                ],
                [vc_output1, vc_output2],
            )
   with gr.TabItem("Download Model"):
            with gr.Row():
                url=gr.Textbox(label="Enter the URL to the Model:")
            with gr.Row():
                model = gr.Textbox(label="Name your model:")
                download_button=gr.Button("Download")
            with gr.Row():
                status_bar=gr.Textbox(label="")
                download_button.click(fn=download_from_url, inputs=[url, model], outputs=[status_bar])
            with gr.Row():
                gr.Markdown(
            )

app.launch(share=True)
