using Documenter
using RATiLQR

format =
Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
)

makedocs(
    format = format,
    sitename = "RATiLQR.jl",
    authors = "Haruki Nishimura",
    source = "source",
    build = "build",
    clean = true,
    doctest = true,
    modules = [RATiLQR],
    pages = ["Home" => "index.md",
             "getting-started.md",
             "optimal-control.md",
             "solvers.md",
             "references.md"]
)

deploydocs(
    repo = "github.com/StanfordMSL/RATiLQR.jl"
)
