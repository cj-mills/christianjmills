# Themes

The site's colours come from GPUI Kit theme files, mapped to the site's colour roles and corrected
by `tools/design/theme.ts`. `tools/design/tokens.ts` writes the result to `../tokens.json`.

- `ayu.json`: GPUI Kit's `themes/ayu.json`, unchanged, from
  `github.com/longbridge/gpui-kit` at commit `b2aaa3e5f9a387dea74981586bfca33e7e08420d`.
  Ayu Light and Ayu Dark are the site's default themes.
  - GPUI Kit is licensed under the Apache License 2.0: `LICENSE-gpui-kit.txt`.
  - The file names Zed Industries as its author and Zed's `assets/themes/ayu` as its source,
    whose licence is MIT, copyright (c) 2016 Ike Ku: `LICENSE-ayu.txt` (from
    `github.com/zed-industries/zed` at commit `e62dd2a0e584154886ff86cc1e9e3e060558b977`).

To use another theme, add its GPUI Kit file here with its licences and change `THEME` in
`tools/design/tokens.ts`.
