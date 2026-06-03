from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    voxcpm_model_path: str = "openbmb/VoxCPM2"
    voxcpm_data_dir: str = "./data"
    voxcpm_devices: str = "0"
    voxcpm_host: str = "0.0.0.0"
    voxcpm_port: int = 8000

    @property
    def device_list(self) -> list[int]:
        return [int(d.strip()) for d in self.voxcpm_devices.split(",")]


settings = Settings()
