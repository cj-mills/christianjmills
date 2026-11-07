// Third-party notices for what every page ships besides the site's own MIT code: the fonts (OFL)
// and the colours derived from the default theme (MIT, through GPUI Kit's Apache-2.0 theme file).
import type { APIRoute } from 'astro';
import { FONTS, packageLicense, packageVersion } from '../../tools/design/fonts.ts';
import ayuLicense from '../styles/themes/LICENSE-ayu.txt?raw';
import gpuiKitLicense from '../styles/themes/LICENSE-gpui-kit.txt?raw';
const rule = '='.repeat(72);

export const GET: APIRoute = () => {
  const parts = [
    'Third-party material served by christianjmills.com',
    '',
    "The site's own code is MIT licensed (the repository's LICENSE). Each deliverable carries its own",
    'licences in its manifest. This file covers what every page uses besides those.',
    '',
    ...FONTS.flatMap((f) => [rule, `Font: ${f.family}, from ${f.package}@${packageVersion(f.package)} (Latin subsets)`, rule, packageLicense(f.package).trim(), '']),
    rule,
    'Colours: derived from the Ayu theme (Ayu Light, Ayu Dark), as packaged by Zed Industries',
    'and redistributed in GPUI Kit (github.com/longbridge/gpui-kit) themes/ayu.json',
    rule,
    ayuLicense.trim(),
    '',
    rule,
    'GPUI Kit, Apache License 2.0',
    rule,
    gpuiKitLicense.trim(),
    '',
  ];
  return new Response(parts.join('\n'), { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
};
