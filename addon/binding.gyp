{
  "targets": [
    {
      "target_name": "financialAddon",
      "sources": [
        "addon.cpp",
        "../cpp_engine/src/Portfolio.cpp",
        "../cpp_engine/src/DataManager.cpp",
        "../cpp_engine/src/PortfolioManager.cpp"
      ],
      "include_dirs": [
        "<!(node -p \"require('node-addon-api').include\")",
        "../cpp_engine/include"
      ],
      "dependencies": [],
      "cflags_cc!": [ "-fno-exceptions" ],
      "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ]
    }
  ]
}