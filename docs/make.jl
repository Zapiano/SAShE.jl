using Documenter
using SAShE

DocMeta.setdocmeta!(SAShE, :DocTestSetup, :(using SAShE); recursive=true)

makedocs(;
    modules=[SAShE],
    authors="Pedro Ribeiro de Almeida",
    sitename="SAShE.jl",
    checkdocs=:exports,
    format=Documenter.HTML(;
        canonical="https://Zapiano.github.io/SAShE.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Concepts" => "concepts.md",
        "Estimators" => "estimators.md",
        "Examples" => "examples.md",
        "Deliberate deviations" => "deviations.md",
        "API reference" => "api.md",
        "Naming conventions" => "naming_conventions.md",
        "References" => "references.md",
    ],
)

deploydocs(; repo="github.com/Zapiano/SAShE.jl", devbranch="main")
