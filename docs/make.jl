using Documenter
using SAShE

DocMeta.setdocmeta!(SAShE, :DocTestSetup, :(using SAShE); recursive=true)

makedocs(;
    modules=[SAShE],
    authors="Pedro Ribeiro de Almeida",
    sitename="SAShE.jl",
    # No API-reference page yet (see docs/src/next_steps.md); re-enable once it exists.
    checkdocs=:none,
    format=Documenter.HTML(;
        canonical="https://Zapiano.github.io/SAShE.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "How it works" => "how_it_works.md",
        "Next steps" => "next_steps.md",
    ],
)

deploydocs(; repo="github.com/Zapiano/SAShE.jl", devbranch="main")
