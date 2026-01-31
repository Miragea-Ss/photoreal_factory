from photoreal_factory.nodes import (
    PhotorealFactoryLive,
    PhotorealFolderLoader,
    PhotorealImageSaver,
)

NODE_CLASS_MAPPINGS = {
    "PhotorealFactoryLive": PhotorealFactoryLive,
    "PhotorealFolderLoader": PhotorealFolderLoader,
    "PhotorealImageSaver": PhotorealImageSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotorealFactoryLive": "🏭 Photoreal Factory (PRO)",
    "PhotorealFolderLoader": "📂 Factory Folder Loader",
    "PhotorealImageSaver": "💾 Factory Image Saver",
}