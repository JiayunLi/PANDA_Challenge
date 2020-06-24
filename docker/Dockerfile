From nvcr.io/nvidia/pytorch:19.06-py3

ARG USER_ID
ARG GROUP_ID
# install opensilde
RUN apt-get update --fix-missing && \
 apt-get install -y python3-openslide &&\
 apt-get install -y tmux
# activate conda and install packages
RUN pip install spams seaborn scikit-image scikit-learn && \
conda install -c conda-forge tqdm && \
conda install -c conda-forge ipywidgets && \
conda install opencv pixman=0.36.0 && \
conda install -c pytorch -c fastai fastai && \
conda install -c conda-forge albumentations && \
conda install -c conda-forge nvidia-apex && \
pip install h5py lmdb openslide-python tensorboardX Pillow tensorboard && \
pip install --upgrade jupyter_client && \
pip install imagecodecs && \
pip install tifffile
RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
  groupadd -g ${GROUP_ID} docker-user &&\
  useradd -l -u ${USER_ID} -g docker-user docker-user &&\
  install -d -m 0755 -o docker-user -g docker-user /home/docker-user \
;fi
USER docker-user