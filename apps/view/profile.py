from logging import getLogger

from fastapi import APIRouter, Body, UploadFile, File

from apps.a_common.error import InvalidParamError
from apps.logic.profile import run
from apps.a_common.response import success_response
from apps.a_common.storage import get_filename_without_uuid_prefix, make_file_url, save_as_temp_file

profile_router = APIRouter()
profile_prefix = 'profile'
logger = getLogger(__name__)


@profile_router.post("/", summary="profile上传接口")
async def upload(iter: int = Body(...), net: UploadFile = File(...), data: UploadFile = File(...)):
    if net.filename[-4:] != '.mge' or data.filename[-4:] != '.pkl':
        return InvalidParamError("文件格式不对")
    net_path = save_as_temp_file(net)
    data_path = save_as_temp_file(data)
    out = run(in_net=net_path, in_data=data_path, in_iter=iter)
    return success_response(out)
