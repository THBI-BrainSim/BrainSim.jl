using BrainSim
using Documenter

DocMeta.setdocmeta!(BrainSim, :DocTestSetup, :(using BrainSim); recursive=true)

makedocs(;
    modules=[BrainSim],
    authors="terrypang <terrypang@aliyun.com> and contributors",
    repo="https://github.com/THBI-BrainSim/BrainSim.jl/blob/{commit}{path}#{line}",
    sitename="BrainSim.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://thbi-brainsim.github.io/BrainSim.jl/dev/",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/THBI-BrainSim/BrainSim.jl",
)
