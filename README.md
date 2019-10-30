# Репозиторий с Back End проекта "Генератор Аватарок"

## 🔨 Гайд по установке и запуску

Пререквизиты: на компьютере должен быть установлен pip3 версии не ниже 19.2.3, python версии не ниже 3.7.4, CMake версии не ниже 3.5.1

Склонировать репозиторий:

`git clone https://github.com/NolanRus/avatar-generator`

Установить зависимости:

`pip3 install -r requirements.txt`

Возможно, возникнут проблемы с установкой cv2 и dlib. Их лучше ставить через anaconda.

`conda install -c menpo dlib`
`conda install -c conda-forge opencv`

Скачать датасет:

`https://storage.cloud.google.com/cartoonset_public_files/cartoonset10k.tgz`

И распаковать в корень проекта. После этого в корне должна оказаться папка cartoonset10k

Запустить сервер:

`python3 server.py`