{
  "targets": [
    {
      "target_name": "financialAddon",
      "cflags!": [ "-fno-exceptions" ],
      "include_dirs": [
        "<!(node -p \"require('node-addon-api').include\")",
        "../cpp_engine/include"
      ],
      "dependencies": [
        "../cpp_engine/financial_cpp.gyp:financial_cpp"
      ],
      "defines": [
        "NAPI_DISABLE_CPP_EXCEPTIONS"
      ],
      "sources": [
        "addon.cpp"
      ]
    }
  ]
}