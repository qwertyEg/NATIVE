import yt_dlp

def download_vk(url: str, output_path: str) -> str:
    return _download(url, output_path)

def download_rutube(url: str, output_path: str) -> str:
    return _download(url, output_path)

def _download(url: str, output_path: str) -> str:
    ydl_opts = {
        'format': 'best',
        'outtmpl': f"{output_path}.%(ext)s",
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)