import{t as e}from"./ChronosTimeline-Cns12ZpJ.js";import{E as t,R as n,S as r,X as i,_ as a,_t as o,g as s,ht as c,x as l}from"./modules/shiki-DY0DGJsA.js";import{t as u}from"./slidev/default-DkFhPs4S.js";import{nt as d,rt as f}from"./index-CSXGUKYK.js";var p=`
> ORDERBY start
> DEFAULTVIEW 2014|2022
> HEIGHT 300
> NOTODAY
- [2015] #2a78d6 FCN · fully convolutional | Replace FC with conv + transposed conv · Semantic segmentation
- [2015] #4c9a2a U-Net · skip concat | Symmetric encoder–decoder + skip concat · Semantic segmentation
- [2015] #8e6bbf SegNet · pooling indices | Encoder–decoder with pooling indices · Semantic segmentation
- [2017] #ed7d31 Mask R-CNN · mask head | Faster R-CNN + mask head · Instance segmentation
- [2017] #2a78d6 PSPNet · pyramid pooling | Pyramid pooling module · Semantic segmentation
- [2017] #2a78d6 DeepLab v3 · atrous + ASPP | Atrous convolutions + ASPP · Semantic segmentation
- [2018] #4c9a2a DeepLab v3+ · encoder–decoder | Encoder–decoder + atrous conv · Semantic segmentation
- [2019] #8e6bbf Panoptic FPN · things + stuff | Unified stuff + things · Panoptic segmentation
`,m={__name:`7_unet_modern_seg.md__slidev_41`,setup(m){let{$slidev:h,$nav:g,$clicksContext:_,$clicks:v,$page:y,$renderContext:b,$frontmatter:x}=f();return _.setup(),(f,m)=>{let h=e;return n(),a(u,o(t(c(d)(c(x),40))),{default:i(()=>[m[0]||=s(`h1`,null,`Segmentation Architecture Evolution`,-1),r(h,{source:p}),m[1]||=s(`div`,{class:`text-xs text-gray-400 text-right mt-6`},`Hover for task details · drag to pan · scroll to zoom`,-1),m[2]||=s(`div`,{class:`mt-3 bg-yellow-50 dark:bg-yellow-900 rounded p-3 text-sm`},[s(`p`,null,[s(`strong`,null,`Common pattern`),l(`: All modern segmentation architectures use some form of:`)]),s(`ol`,null,[s(`li`,null,[s(`strong`,null,`Pretrained encoder`),l(` (backbone) for feature extraction`)]),s(`li`,null,[s(`strong`,null,`Decoder`),l(` with upsampling for resolution recovery`)]),s(`li`,null,[s(`strong`,null,`Multi-scale features`),l(` (FPN, ASPP, pyramid pooling)`)]),s(`li`,null,[s(`strong`,null,`Skip connections`),l(` to preserve spatial detail`)])])],-1)]),_:1},16)}}};export{m as default};