from django.core.management.base import BaseCommand
from django.utils.text import slugify

from catalog.models import Dish, DishIngredient, Ingredient


REFERENCE_COCINAR = (
    "https://comococinar.pe/los-100-platos-mas-representativos-de-las-comidas-peruanas-y-sus-recetas/"
)
RECIPES = {
    "Aguadito": (
        "Sopa criolla de arroz y culantro, reconfortante y muy popular en reuniones familiares.",
        "costa",
        "🍲",
        ["pollo", "arroz", "culantro", "papa", "zanahoria", "arvejas", "cebolla", "ajo", "ají amarillo", "choclo"],
        "",
    ),
    "Aji de Gallina": (
        "Guiso cremoso de gallina deshilachada con ají amarillo, servido tradicionalmente con papa y arroz.",
        "costa",
        "🍛",
        ["gallina", "ají amarillo", "pan", "leche evaporada", "pecanas", "cebolla", "ajo", "papa", "arroz", "huevo", "aceituna negra"],
        REFERENCE_COCINAR,
    ),
    "Anticuchos": (
        "Brochetas sazonadas a la parrilla, emblema de las calles y celebraciones limeñas.",
        "costa",
        "🍢",
        ["corazón de res", "ají panca", "ajo", "vinagre", "comino", "orégano", "papa", "choclo"],
        "",
    ),
    "Arroz Chaufa": (
        "Arroz salteado al wok que expresa la fusión chifa peruano-china.",
        "costa",
        "🍚",
        ["arroz", "pollo", "huevo", "cebolla china", "sillao", "ajo", "kion", "aceite"],
        "https://recetascocinaperuana.com/arroz-chaufa/",
    ),
    "Arroz con Mariscos": (
        "Arroz sazonado con ajíes y frutos del mar, muy representativo de la cocina costeña.",
        "costa",
        "🦐",
        ["arroz", "langostinos", "calamar", "pulpo", "conchas", "pescado", "ají amarillo", "ají panca", "cebolla", "ajo", "culantro", "arvejas"],
        "",
    ),
    "Arroz con Pollo": (
        "Arroz verde con pollo, culantro y verduras; suele servirse con salsa criolla.",
        "costa",
        "🍗",
        ["arroz", "pollo", "culantro", "ají amarillo", "cebolla", "ajo", "arvejas", "zanahoria", "choclo", "pimiento", "cerveza"],
        REFERENCE_COCINAR,
    ),
    "Arroz Tapado": (
        "Arroz moldeado que cubre un relleno criollo de carne sazonada y otros acompañamientos.",
        "costa",
        "🍚",
        ["arroz", "carne molida", "cebolla", "ajo", "tomate", "ají panca", "huevo", "aceituna negra", "pasas"],
        "",
    ),
    "Bisteck a lo Pobre": (
        "Plato contundente que reúne bistec, huevo frito, plátano y papas fritas.",
        "costa",
        "🥩",
        ["bistec de res", "huevo", "plátano", "papa", "arroz", "cebolla", "tomate"],
        "",
    ),
    "Caldo de Gallina": (
        "Caldo intenso de gallina con fideos y papa, servido caliente con hierbas y limón.",
        "costa",
        "🍜",
        ["gallina", "fideos", "papa", "huevo", "apio", "kion", "cebolla china", "orégano", "limón"],
        "",
    ),
    "Carapulcra": (
        "Guiso de papa seca con ajíes y carne, asociado a la cocina tradicional de Chincha y el sur.",
        "costa",
        "🥘",
        ["papa seca", "carne de cerdo", "pollo", "ají panca", "maní", "cebolla", "ajo", "comino", "caldo"],
        "",
    ),
    "Cau Cau": (
        "Guiso criollo de mondongo y papa aromatizado con palillo y hierbabuena.",
        "costa",
        "🍲",
        ["mondongo", "papa", "ají amarillo", "palillo", "cebolla", "ajo", "hierbabuena", "arvejas", "zanahoria"],
        "",
    ),
    "Causa Limeña": (
        "Preparación fría en capas de papa amarilla aliñada y relleno, popular en mesas limeñas.",
        "costa",
        "🥔",
        ["papa amarilla", "ají amarillo", "limón", "aceite", "pollo", "palta", "mayonesa", "huevo", "aceituna negra"],
        "https://recetascocinaperuana.com/causa-limena/",
    ),
    "Ceviche": (
        "Pescado fresco marinado en limón con cebolla, ají y culantro; suele acompañarse con camote y choclo.",
        "costa",
        "🐟",
        ["pescado blanco", "limón", "cebolla roja", "ají limo", "culantro", "camote", "choclo", "sal", "lechuga"],
        REFERENCE_COCINAR,
    ),
    "Chanfainita": (
        "Guiso popular de bofe con papa, hierbabuena y ají panca, servido a menudo con mote.",
        "costa",
        "🥘",
        ["bofe de res", "papa", "ají panca", "cebolla", "ajo", "hierbabuena", "comino", "mote"],
        "",
    ),
    "Chicharron de cerdo": (
        "Cerdo cocido y dorado, tradicionalmente servido con camote y salsa criolla.",
        "costa",
        "🍖",
        ["cerdo", "camote", "cebolla roja", "limón", "hierbabuena", "choclo", "sal"],
        "",
    ),
    "Chicharron de Mariscos": (
        "Bocados de mariscos sazonados y fritos, servidos con limón y salsas.",
        "costa",
        "🦑",
        ["calamar", "pulpo", "langostinos", "pescado", "harina", "huevo", "limón", "ajo", "comino"],
        "",
    ),
    "Choclo con Queso": (
        "Choclo tierno acompañado de queso fresco, sencillo y muy conocido en la sierra peruana.",
        "sierra",
        "🌽",
        ["choclo", "queso fresco", "sal"],
        "",
    ),
    "Choritos a la Chalaca": (
        "Mejillones frescos cubiertos con una salsa cruda de cebolla, tomate, ají y limón.",
        "costa",
        "🐚",
        ["choros", "cebolla roja", "tomate", "ají limo", "limón", "choclo", "culantro", "sal"],
        "",
    ),
    "Escabeche": (
        "Preparación criolla de proteína con cebolla, ají y vinagre; existen versiones de pescado y de pollo.",
        "diversas",
        "🫑",
        ["pollo", "pescado", "cebolla roja", "ají amarillo", "ají panca", "vinagre", "ajo", "comino", "orégano", "huevo", "aceituna negra"],
        "https://recetascocinaperuana.com/escabeche-de-pescado/",
    ),
    "Frejoles": (
        "Menestra casera de frejoles con aderezo criollo, servida con arroz y a veces carne de cerdo.",
        "diversas",
        "🫘",
        ["frejoles", "cebolla", "ajo", "ají panca", "comino", "tocino", "arroz"],
        "",
    ),
    "Humitas": (
        "Masa suave de choclo molido envuelta en sus hojas, preparada en versiones dulces y saladas.",
        "sierra",
        "🌽",
        ["choclo", "queso fresco", "cebolla", "mantequilla", "leche", "anís", "sal"],
        "",
    ),
    "Lentejitas": (
        "Guiso hogareño de lentejas con aderezo y verduras, normalmente acompañado de arroz.",
        "diversas",
        "🥣",
        ["lentejas", "cebolla", "ajo", "zanahoria", "tomate", "comino", "arroz"],
        "",
    ),
    "Locro": (
        "Guiso espeso de zapallo y papa con hierbas, queso y otros productos de temporada.",
        "sierra",
        "🎃",
        ["zapallo", "papa", "choclo", "arvejas", "habas", "queso fresco", "leche", "huacatay", "cebolla"],
        "",
    ),
    "Lomo Saltado": (
        "Salteado criollo de res, cebolla y tomate con sillao, acompañado de papas fritas y arroz.",
        "costa",
        "🥩",
        ["lomo de res", "cebolla roja", "tomate", "ají amarillo", "sillao", "vinagre", "papa", "arroz", "culantro"],
        REFERENCE_COCINAR,
    ),
    "Ocopa": (
        "Papas cubiertas con una salsa arequipeña de huacatay, ají mirasol, queso y maní.",
        "sierra",
        "🥔",
        ["papa", "huacatay", "ají mirasol", "maní", "queso fresco", "leche evaporada", "galletas", "ajo", "cebolla", "huevo", "aceituna negra"],
        "https://recetascocinaperuana.com/ocopa-arequipena-receta-peruana/",
    ),
    "Olluquito": (
        "Guiso andino de olluco en tiras con carne o charqui, servido frecuentemente con arroz.",
        "sierra",
        "🥘",
        ["olluco", "carne de res", "charqui", "ají panca", "ají amarillo", "cebolla", "ajo", "comino", "perejil", "arroz"],
        "https://recetascocinaperuana.com/olluquito-peruano/",
    ),
    "Papa a la Huancaina": (
        "Papas sancochadas con una crema de ají amarillo y queso, servidas sobre hojas de lechuga.",
        "sierra",
        "🥔",
        ["papa", "ají amarillo", "queso fresco", "leche evaporada", "galletas", "aceite", "huevo", "aceituna negra", "lechuga"],
        REFERENCE_COCINAR,
    ),
    "Papa Rellena": (
        "Masa de papa rellena con un aderezo de carne, formada y dorada hasta quedar crocante.",
        "costa",
        "🥔",
        ["papa", "carne molida", "cebolla", "ajo", "huevo", "aceituna negra", "pasas", "harina", "comino"],
        "",
    ),
    "Patasca": (
        "Sopa serrana de cocción lenta con mote y carne, servida caliente y con hierbas.",
        "sierra",
        "🍲",
        ["mote", "carne de res", "mondongo", "pata de res", "hierbabuena", "cebolla", "ajo", "ají panca"],
        "",
    ),
    "Picante de cuy": (
        "Preparación andina de cuy con ajíes y especias, servida usualmente con papa.",
        "sierra",
        "🐹",
        ["cuy", "papa", "ají panca", "ají amarillo", "maní", "ajo", "cebolla", "comino", "huacatay"],
        "",
    ),
    "Pollo a la Brasa": (
        "Pollo marinado con especias y asado, uno de los platos más populares del Perú.",
        "costa",
        "🍗",
        ["pollo", "sillao", "vinagre", "ají panca", "ajo", "comino", "orégano", "pimienta", "mostaza", "papas"],
        "https://recetascocinaperuana.com/pollo-a-la-brasa-receta-peruana/",
    ),
    "Pollo al horno con puré": (
        "Pollo horneado sazonado con hierbas, servido con puré de papa cremoso.",
        "costa",
        "🍗",
        ["pollo", "ajo", "comino", "orégano", "limón", "papa", "leche", "mantequilla"],
        "https://recetascocinaperuana.com/pollo-al-horno/",
    ),
    "Pulpo al Olivo": (
        "Láminas de pulpo cocido con una salsa cremosa de aceitunas negras de botija.",
        "costa",
        "🐙",
        ["pulpo", "aceituna negra de botija", "mayonesa", "queso fresco", "leche evaporada", "limón", "lechuga", "huevo"],
        "https://recetascocinaperuana.com/pulpo-al-olivo/",
    ),
    "Rocoto Relleno": (
        "Rocoto arequipeño relleno de carne y especias, cubierto con queso y horneado.",
        "sierra",
        "🌶️",
        ["rocoto", "carne molida", "cebolla", "ajo", "ají panca", "tomate", "huevo", "aceituna negra", "pasas", "queso fresco", "leche"],
        "https://recetascocinaperuana.com/rocoto-relleno/",
    ),
    "Sopa a la Minuta": (
        "Sopa rápida de carne y fideos con leche y huevo, típica de la cocina casera.",
        "costa",
        "🍜",
        ["carne molida", "fideos", "leche evaporada", "huevo", "tomate", "cebolla", "ajo", "ají panca", "orégano"],
        "",
    ),
    "Tacu Tacu": (
        "Tortilla dorada de arroz y menestra, nacida como receta de aprovechamiento criolla.",
        "costa",
        "🍳",
        ["arroz", "frejoles", "cebolla", "ajo", "ají amarillo", "ají panca", "comino", "huevo", "bistec"],
        "https://recetascocinaperuana.com/tacu-tacu-de-pallares/",
    ),
    "Tallarin Saltado": (
        "Fideos salteados al wok con verduras, sillao y carne o pollo, de tradición chifa.",
        "costa",
        "🍝",
        ["tallarines", "carne de res", "pollo", "cebolla", "tomate", "sillao", "cebolla china", "ají amarillo", "ajo"],
        "https://recetascocinaperuana.com/tallarin-chino-peruano/",
    ),
    "Tallarines Rojos": (
        "Pasta con salsa roja criolla de tomate y ají panca, comúnmente servida con pollo.",
        "costa",
        "🍝",
        ["tallarines", "pollo", "tomate", "ají panca", "cebolla", "ajo", "zanahoria", "laurel", "queso parmesano"],
        "https://recetascocinaperuana.com/tallarin-rojo-con-pollo/",
    ),
    "Tallarines Verdes": (
        "Pasta con salsa verde de albahaca y espinaca, queso y leche; suele acompañarse con bistec.",
        "costa",
        "🍝",
        ["tallarines", "albahaca", "espinaca", "queso fresco", "leche evaporada", "maní", "ajo", "queso parmesano", "bistec"],
        "https://recetascocinaperuana.com/tallarin-verde-peruano/",
    ),
    "Tamales": (
        "Masa de maíz sazonada y rellena, envuelta en hojas y cocida al vapor.",
        "diversas",
        "🫔",
        ["maíz", "carne de cerdo", "pollo", "ají panca", "cebolla", "ajo", "huevo", "aceituna", "maní", "hojas de plátano"],
        "",
    ),
}

OPTIONAL_INGREDIENT_NOTES = {
    "Arroz con Pollo": {
        "cerveza": "Opcional; algunas recetas la usan para el aderezo.",
    },
    "Ceviche": {
        "camote": "Acompañamiento habitual, pero no forma parte del ceviche en sí.",
        "choclo": "Acompañamiento habitual, pero no forma parte del ceviche en sí.",
        "lechuga": "Acompañamiento opcional; puede variar según la presentación.",
    },
    "Escabeche": {
        "pollo": "Alternativa al pescado; las versiones suelen elegir una sola proteína.",
        "pescado": "Alternativa al pollo; las versiones suelen elegir una sola proteína.",
    },
    "Frejoles": {
        "tocino": "Opcional; algunas versiones incorporan tocino o carne de cerdo.",
    },
    "Humitas": {
        "queso fresco": "Se usa sobre todo en versiones saladas; el relleno cambia según la variante.",
        "cebolla": "Se usa en algunas versiones saladas; no es común a todas las humitas.",
        "anís": "Se usa en algunas versiones dulces; no es común a todas las humitas.",
    },
    "Olluquito": {
        "carne de res": "Alternativa al charqui; la receta suele elegir una proteína.",
        "charqui": "Alternativa a la carne de res; la receta suele elegir una proteína.",
    },
    "Tallarin Saltado": {
        "carne de res": "Proteína a elección; puede prepararse con carne o pollo.",
        "pollo": "Proteína a elección; puede prepararse con carne o pollo.",
    },
    "Tallarines Rojos": {
        "pollo": "Acompañamiento habitual; también hay versiones con otras proteínas o sin carne.",
    },
    "Tallarines Verdes": {
        "bistec": "Acompañamiento opcional; el tallarín verde también se sirve solo o con otra proteína.",
    },
    "Tacu Tacu": {
        "bistec": "Acompañamiento opcional; el tacu tacu también se sirve solo o con otros acompañamientos.",
    },
    "Tamales": {
        "carne de cerdo": "Relleno variable; algunas versiones llevan cerdo y otras pollo.",
        "pollo": "Relleno variable; algunas versiones llevan pollo y otras cerdo.",
    },
}


class Command(BaseCommand):
    help = "Carga o actualiza el catálogo inicial de los 40 platos PeruFoodNet."

    def handle(self, *args, **options):
        if len(RECIPES) != 40:
            raise RuntimeError(f"Se esperaban 40 platillos y hay {len(RECIPES)}.")

        relations = 0
        for name, (description, region, emoji, ingredient_names, source_url) in RECIPES.items():
            dish, _ = Dish.objects.update_or_create(
                name=name,
                defaults={
                    "slug": slugify(name),
                    "description": description,
                    "region": region,
                    "emoji": emoji,
                },
            )
            optional_ingredients = OPTIONAL_INGREDIENT_NOTES.get(name, {})
            for ingredient_name in ingredient_names:
                ingredient, _ = Ingredient.objects.get_or_create(name=ingredient_name)
                DishIngredient.objects.update_or_create(
                    dish=dish,
                    ingredient=ingredient,
                    defaults={
                        "is_optional": ingredient_name in optional_ingredients,
                        "notes": optional_ingredients.get(
                            ingredient_name,
                            "Ingrediente habitual de referencia; las recetas pueden variar.",
                        ),
                        "source_url": source_url,
                    },
                )
                relations += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Catálogo cargado: {len(RECIPES)} platillos y {relations} asociaciones de ingredientes."
            )
        )
        self.stdout.write(
            "La lista es orientativa; verifica ingredientes y fuentes antes de usarla para decisiones alimentarias."
        )
