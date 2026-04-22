# Стиль puml диаграмм

Документ фиксирует стиль существующих диаграмм в `diagrams/puml` и задает правила генерации новых PlantUML-диаграмм для `gendoc`.

## Какие типы диаграмм уже есть

В текущем наборе диаграмм фактически используется 3 уровня представления.

### 1. Главная диаграмма одного роута

Это основная `activity`-диаграмма бизнес-роута.

Примеры:

- `DG-1-R.GetRetrieveSteamOrderById.activity.puml`
- `DG-2-R.GetRetrieveTopUpGames.activity.puml`
- `DG-3-R.GetRetrieveVouchers.activity.puml`

Такая диаграмма описывает:

- входные параметры;
- ключевые действия;
- обращения к БД;
- обращения во внешние системы;
- ветвления и бизнес-правила;
- формирование ответа.

### 2. Поддиаграмма отдельного шага

Это отдельная `activity`-диаграмма для одного шага из основной диаграммы.

Примеры:

- `DG-2-R.GetRetrieveTopUpGames.GetDonateHubProducts.activity.puml`
- `DG-1-R.GetRetrieveSteamAmount.GetWataApiRates.activity.puml`
- `DG-2-R.GetRetrieveTopUpGames.TransformPositionsByMappingAndCalculate.activity.puml`

Такая диаграмма используется, когда один шаг основной схемы нужно детализировать:

- внешний интеграционный вызов;
- transform/mapping step;
- calculate step;
- цикл `foreach`;
- отдельный helper/usecase.

### 3. Data-transformation диаграмма

Это отдельное представление того же роута, но через поток данных и преобразования.

Примеры:

- `DG-2-R.GetRetrieveTopUpGames.data-transformation.puml`
- `DG-2-R.GetRetrieveTopUpGames.data-transformation.full.puml`

Такая диаграмма показывает:

- входные коллекции;
- промежуточные структуры;
- `map/filter/group by`;
- связи между стадиями;
- итоговый результат.

## Есть ли диаграммы не одного роута

В текущем наборе нет диаграмм, которые описывают список роутов сразу.

Есть только:

- одна диаграмма одного роута;
- дочерние диаграммы шагов этого же роута;
- альтернативный `data-transformation` view этого же роута.

Вывод:

- единица генерации должна быть `один route`;
- внутри route можно порождать дочерние диаграммы;
- `data-transformation` является отдельным view того же route, а не отдельным route.

## На какие блоки стоит разбивать route

### 1. `route-orchestration`

Главная `activity`-диаграмма endpoint-а.

В неё входит:

- request/body/path/query;
- ключевые интеграции;
- важные проверки;
- DB-шаги;
- build response.

### 2. `integration-step`

Отдельная `activity`-диаграмма на один внешний вызов.

Подходит для:

- `GET/POST/...` во внешний API;
- обработки `Response.Code`;
- разветвлённой логики статусов;
- request/response контракта интеграции.

### 3. `transform-step`

Отдельная `activity`-диаграмма для локального преобразования.

Подходит для:

- `map dto -> dto`;
- `foreach`;
- `filter`;
- `group by`;
- `calculate`.

### 4. `data-transformation`

Отдельная диаграмма потока данных.

Нужна, когда route имеет:

- несколько коллекций;
- промежуточные map/list/set объекты;
- длинный pipeline преобразований;
- вложенные mapping/calculation блоки.

## Когда route нужно дробить

Route стоит дробить на поддиаграммы, если внутри есть хотя бы одно из:

- больше двух внешних интеграций;
- цикл `foreach`;
- явный DTO mapping;
- отдельный calculate step;
- сложный status/lifecycle flow;
- длинная диаграмма более чем на 25-35 действий.

## Основная нотация в activity-диаграммах

### Вход и выход

Типичный старт:

```puml
:Request;
note: R1
```

Типичное завершение:

```puml
:Build response\nwith ...;
note: R2

:HTTP 200 OK;
```

Семантика:

- `R1` — вход;
- `R2` — выход.

### Переменные

Переменные задаются как action-блоки:

```puml
:**orderId** = Request.Params.id;
:**terminalPublicId** = **terminal.publicId**;
:**voucher** : VoucherDto = map (**donateHubVoucher**);
```

Типовые шаблоны:

- `:**x** = ...;`
- `:**x** : Type = ...;`

Особенности:

- важные сущности выделяются через `**bold**`;
- типы пишутся прямо в тексте;
- используются формы `list<T>`, `map<key, value>`, `Dto`.

### Внешние вызовы

Внешние вызовы обычно помечаются зеленым:

```puml
#palegreen :**terminal** = retrieve\nby GET /api/h2h/terminals\nfrom WataApi;
note: A1
```

Типовые глаголы:

- `retrieve`
- `create`
- `transform`
- `calculate`
- `map`
- `filter`

### Работа с БД

Работа с БД оформляется текстовыми шагами внутри `partition`, а не через UML database-объекты.

Типовые шаблоны:

```puml
partition Найти заказ в БД {
    if (find in DB `Orders` where (...)) then (-)
```

```puml
:create in DB `Orders` with (...);
```

```puml
:update in DB `Orders` where (...) on (...);
```

Рекомендуемые стандартные формулировки:

- `find in DB \`Table\` where (...)`
- `create in DB \`Table\` with (...)`
- `update in DB \`Table\` where (...) on (...)`

### Notes и комментарии

`note` используется очень активно и является частью стиля.

Примеры:

```puml
note: A1
```

```puml
note
    Идентификатор заказа
    в системе Пользователя
end note
```

```puml
note left: A3
note right of R2_result: R2
floating note
    ...
end note
```

Через `note` обычно:

- помечается код шага `A1`, `M1`, `C1`, `R2`;
- поясняется смысл переменной;
- фиксируется бизнес-логика;
- описывается причина ветки или ошибки.

### Ветвления

Типичный формат:

```puml
if (**Response.Code** = HTTP 200?) then (+)
...
endif
```

Также активно используются:

- вложенные `if`;
- ранний `end`;
- ветки по статусам;
- отдельные ветки ошибок.

### Циклы

Типичный цикл:

```puml
repeat :foreach **item** in **items**;
...
repeat while;
```

Этот паттерн хорошо подходит для генерации шагов:

- `map`;
- `calculate`;
- добавление элементов в результирующую коллекцию.

## Семантические коды шагов

В диаграммах уже используется устойчивая система кодов:

- `A*` — action / integration step;
- `M*` — mapping;
- `C*` — calculation;
- `R*` — request/response.

Примеры:

- `A1`
- `A2.R2`
- `M1`
- `C1`
- `R1`
- `R2`

Для генератора это стоит сохранить как внутреннюю классификацию узлов.

## Цвета

На практике в текущем наборе есть 2 заметных цвета.

### `#palegreen`

Используется для:

- внешних API-вызовов;
- create/retrieve интеграций;
- иногда для важных `transform` или `calculate` шагов.

### `#GhostWhite`

Используется в `data-transformation` диаграммах для крупных transformation-блоков.

### Рекомендация для генератора

Для MVP достаточно:

- без цвета для обычных локальных действий;
- `#palegreen` для внешних вызовов и заметных вычислительных шагов;
- `#GhostWhite` только для data-transformation блоков.

## Типичные формулировки действий

### Request / params

```puml
:**orderId** = Request.Params.id;
:**serviceId** = Request.Query.serviceId;
:**amount** = Request.Body.amount;
```

### Transform / map / calculate

```puml
:**voucher** : VoucherDto = map (**donateHubVoucher**);
#palegreen :**totalAmountRub** = calculate(...);
:**filteredPositionsGroupedByProductId** = transform (...);
```

### Response

```puml
:Build response\nusing **topUpGames**;
```

или

```puml
:Build response\nwith `orderId` = **orderId**, ...;
```

### Ошибки

```puml
:HTTP 400 BadRequest\nwith error code **Config.ErrorCodes.NotFound**;
```

## Практическое правило генерации

Для одного route генератор должен уметь строить 4 класса шагов.

### `R`

- request params;
- body;
- build response.

### `A`

- external API call;
- integration helper;
- create/retrieve во внешнем сервисе.

### `DB`

- `find/create/update in DB`.

### `M/C`

- `map`;
- `transform`;
- `calculate`;
- `filter`;
- `group by`.

## Рекомендация по генерации route

В MVP стоит делать следующее:

- всегда генерировать одну `main activity`-диаграмму на route;
- генерировать дочернюю `integration activity`, если внешний вызов имеет разветвление по `Response.Code`;
- генерировать дочернюю `transform activity`, если внутри есть `foreach/map/calculate`;
- генерировать `data-transformation`, если route представляет собой выраженный pipeline над коллекциями.

## Краткий итог

Текущий стиль диаграмм в репозитории — это не строгий “академический UML”, а практический business-flow стиль поверх PlantUML.

Главные свойства этого стиля:

- narrative activity flow;
- переменные как action-блоки;
- богатое использование `note`;
- текстовая запись DB-действий;
- зеленый цвет для внешних вызовов;
- декомпозиция route на поддиаграммы шагов;
- отдельный `data-transformation` view для сложных pipeline.

Именно этот стиль нужно воспроизводить в `gendoc`, а не пытаться заменять его на “чистый UML”.
